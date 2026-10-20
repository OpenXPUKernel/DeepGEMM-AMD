#pragma once
#include <deep_gemm/impls/cdna4/mega_moe.cuh>
namespace moe_pointwise {
__device__ __forceinline__ void output_pointers(const MegaMoeArgs& p) {
  using namespace mega_moe;
  const auto& l=p.layout;
  extern __shared__ int scratch[];
  uni_symm::device::fence_acquire<Scope::system>();
  const Rows rows=launch_rows(p,scratch);
  for(int row=rows.begin+blockIdx.x*blockDim.x+threadIdx.x;row<rows.count;row+=gridDim.x*blockDim.x) {
    if(!rows.valid(row))continue;
    uint32_t source=region<const uint32_t>(p,l.l1_records)[int64_t(row)*l.record_ints+l.hidden/128+1];
    const int slot=source%l.num_topk,token=source/l.num_topk%l.num_max_tokens,rank=source/l.num_topk/l.num_max_tokens;
    region<int64_t>(p,l.output_rows)[row]=reinterpret_cast<int64_t>(p.view.map(
      region<uint16_t>(p,l.combine)+(int64_t(slot)*l.num_max_tokens+token)*l.hidden,rank));
  }
}

template<int H,int TopK>
__device__ __forceinline__ void reduce(const MegaMoeArgs& p) {
#if DG_MOE_COHERENT_REDUCE
  // For these row widths a wave never straddles tokens. Load its validity mask
  // once, and use scalar buffer bases plus 32-bit offsets for the payload.
  if constexpr(H%512==0 && TopK<=32) {
    using namespace mega_moe;
    const uint64_t plane=uint64_t(p.layout.num_max_tokens)*H*2;
    const uint64_t output_bytes=uint64_t(p.num_tokens)*H*2;
    if(plane*TopK<=0xffffffffu && output_bytes<=0xffffffffu) {
      // Peer stores are already released by the preceding cross-rank barrier.
      // SC1 payload reads use system coherence without per-workgroup invalidation.
      asm volatile("" ::: "memory");
      const auto* ids=region<const int64_t>(p,p.layout.topk_idx);
      const auto input=make_rsrc(region<const uint8_t>(p,p.layout.combine),uint32_t(plane*TopK));
      const auto output=make_rsrc(p.y,uint32_t(output_bytes));
      const auto shared=make_rsrc(p.shared_out,p.shared_out?uint32_t(output_bytes):0u);
      for(uint32_t linear=blockIdx.x*blockDim.x+threadIdx.x;
          linear<uint32_t(p.num_tokens)*(H/8);linear+=gridDim.x*blockDim.x) {
        const int token=__builtin_amdgcn_readfirstlane(linear/(H/8));
        unsigned mask=0;
#pragma unroll
        for(int slot=0;slot<TopK;++slot)
          mask|=unsigned(ids[int64_t(token)*TopK+slot]>=0)<<slot;
        v4i values[TopK];
#pragma unroll
        for(int slot=0;slot<TopK;++slot)
          values[slot]=(mask&(1u<<slot))?__builtin_amdgcn_raw_buffer_load_b128(input,linear*16,uint32_t(slot*plane),16):v4i{};
        float sum[8]={};
#pragma unroll
        for(int slot=0;slot<TopK;++slot) {
#pragma unroll
          for(int i=0;i<4;++i){float a,b;unpack_bf16x2(uint32_t(values[slot][i]),a,b);sum[2*i]+=a;sum[2*i+1]+=b;}
        }
        if(p.shared_out) {
          const v4i value=__builtin_amdgcn_raw_buffer_load_b128(shared,linear*16,0,0);
#pragma unroll
          for(int i=0;i<4;++i){float a,b;unpack_bf16x2(uint32_t(value[i]),a,b);sum[2*i]+=a;sum[2*i+1]+=b;}
        }
        v4u packed;
#pragma unroll
        for(int i=0;i<4;++i)packed[i]=pack_bf16x2(sum[2*i],sum[2*i+1]);
        buffer_store16<0>(packed,output,linear*16,0);
      }
      return;
    }
  }
#endif
#if DG_MOE_VECTOR_REDUCE
  using namespace mega_moe;
  const auto& l=p.layout;
  const auto* ids=region<const int64_t>(p,l.topk_idx);
  const auto* input=region<const uint16_t>(p,l.combine);
  uni_symm::device::fence_acquire<Scope::system>();
  for(int64_t linear=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
      linear<int64_t(p.num_tokens)*(H/8);linear+=int64_t(gridDim.x)*blockDim.x) {
    const int t=linear/(H/8),h=(linear%(H/8))*8;
    v4i values[TopK];
#pragma unroll
    for(int slot=0;slot<TopK;++slot) {
      const bool valid=ids[int64_t(t)*TopK+slot]>=0;
      values[slot]=valid?*reinterpret_cast<const v4i*>(input+(int64_t(slot)*l.num_max_tokens+t)*H+h):v4i{};
    }
    __builtin_amdgcn_sched_barrier(0);
    float acc[8]={};
#pragma unroll
    for(int slot=0;slot<TopK;++slot) {
#pragma unroll
      for(int i=0;i<4;++i) {
        float a,b;unpack_bf16x2(uint32_t(values[slot][i]),a,b);
        acc[2*i]+=a;acc[2*i+1]+=b;
      }
    }
    if(p.shared_out) {
      const v4i v=*reinterpret_cast<const v4i*>(p.shared_out+int64_t(t)*H+h);
#pragma unroll
      for(int i=0;i<4;++i) {
        float a,b;unpack_bf16x2(uint32_t(v[i]),a,b);acc[2*i]+=a;acc[2*i+1]+=b;
      }
    }
    v4i out;
#pragma unroll
    for(int i=0;i<4;++i)out[i]=pack_bf16x2(acc[2*i],acc[2*i+1]);
    *reinterpret_cast<v4i*>(p.y+linear*8)=out;
  }
#else
  mega_moe_reduce_impl(p);
#endif
}

}
