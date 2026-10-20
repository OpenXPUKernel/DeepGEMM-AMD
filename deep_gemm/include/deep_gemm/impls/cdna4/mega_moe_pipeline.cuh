#pragma once
#include <deep_gemm/impls/cdna4/mega_moe_pipeline_gemm.cuh>
#include <deep_gemm/impls/cdna4/mega_moe_pipeline_epilogue.cuh>
#include <deep_gemm/impls/cdna4/mega_moe_pipeline_memory.cuh>
#include <deep_gemm/impls/cdna4/mega_moe.cuh>
#include <deep_gemm/impls/cdna4/mega_moe_pipeline_args.h>
#ifndef DG_MOE_NATIVE_CORE
#define DG_MOE_NATIVE_CORE 0
#endif
#ifndef DG_MOE_NATIVE_SHARED
#define DG_MOE_NATIVE_SHARED 0
#endif
#if DG_MOE_NATIVE_CORE
#include <deep_gemm/impls/cdna4/mega_moe_pipeline_gemm.cuh>
#endif

#ifndef DG_MOE_COMM_SF
#define DG_MOE_COMM_SF 0
#endif

#ifndef DG_MOE_TASK_M
#define DG_MOE_TASK_M 128
#endif

#ifndef DG_MOE_BLOCK_PIPELINE
#define DG_MOE_BLOCK_PIPELINE 0
#endif
#ifndef DG_MOE_STREAM_K
#define DG_MOE_STREAM_K 0
#endif

#ifndef DG_MOE_PUSH_BLOCKS
#define DG_MOE_PUSH_BLOCKS 0
#endif
#ifndef DG_MOE_SHARED_SPLIT
#define DG_MOE_SHARED_SPLIT 0
#endif
#ifndef DG_MOE_SHARED_TAIL
#define DG_MOE_SHARED_TAIL 0
#endif
#ifndef DG_MOE_EARLY_READY
#define DG_MOE_EARLY_READY 0
#endif
#ifndef DG_MOE_STAGE1_ONLY
#define DG_MOE_STAGE1_ONLY 0
#endif
#ifndef DG_MOE_SEND_ROWS
#define DG_MOE_SEND_ROWS 64
#endif
#ifndef DG_MOE_FAST_INIT
#define DG_MOE_FAST_INIT 0
#endif
#ifndef DG_MOE_L1_ONLY_PUBLISH
#define DG_MOE_L1_ONLY_PUBLISH 0
#endif
#ifndef DG_MOE_L1_LOCAL_ONLY
#define DG_MOE_L1_LOCAL_ONLY 0
#endif
#ifndef DG_MOE_EARLY_POINTERS
#define DG_MOE_EARLY_POINTERS 0
#endif
namespace nv_pipeline {
constexpr int RowM = DG_MOE_TASK_M;
using namespace deep_gemm::cdna4;
using namespace uni_symm::device;
using namespace mega_moe;
using nv_pipeline_detail::coherent_load16;
using nv_pipeline_detail::coherent_store16;
enum { NextL1, HeadL2, TailL2, DoneL2, NumBlocks, NumL1, NumL2, NumCombine, HeadCombine, TailCombine, HeadInput, HeadPush, NumPush, CommDone, CacheEpoch, StateWords = 544 };
__device__ __forceinline__ int* input_flags(const MegaMoeArgs& p) {
#if DG_MOE_EARLY_POINTERS
  return region<int>(p,DG_MOE_INPUT_FLAG_OFFSET);
#else
  return region<int>(p,p.layout.output_rows);
#endif
}

__device__ __forceinline__ void wait_local(const int* flag, int target) {
  if (threadIdx.x == 0) {
    const auto start = clock_ns();
    while (load_relaxed<Scope::device>(flag) < target) {
      __builtin_amdgcn_s_sleep(8);
      if (clock_ns() - start > 60000000000ull) { printf("MoE queue local wait timeout\n"); trap(); }
    }
    (void)load_relaxed<Scope::device>(flag);
  }
  __syncthreads();
  asm volatile("" ::: "memory");
}

// Stable block-wide prefix: four waves, no atomics. It preserves exactly the
// original expert/destination order while parallelizing metadata construction.
__device__ __forceinline__ int prefix256(int value,int* wave_totals,int& total) {
  const int lane=threadIdx.x%64,wave=threadIdx.x/64;
  int inclusive=value;
#pragma unroll
  for(int offset=1;offset<64;offset*=2){const int other=__shfl_up(inclusive,offset,64);if(lane>=offset)inclusive+=other;}
  if(lane==63)wave_totals[wave]=inclusive;
  __syncthreads();
  int carry=0;total=0;
  for(int w=0;w<4;++w){if(w<wave)carry+=wave_totals[w];total+=wave_totals[w];}
  const int before=carry+inclusive-value;
  __syncthreads();
  return before;
}

__device__ __forceinline__ void parallel_init_stage1(const MegaMoePipelineArgs& q) {
  __shared__ int wave_totals[4];
  const auto& p=q.moe;const auto& l=p.layout;const int tid=threadIdx.x;
  const auto* ends=region<const int>(p,l.psum_ends);
  int first=0,end=0,count=0;
  if(tid<l.num_experts_per_rank){first=expert_start(ends,tid,l.alignment);end=ends[tid];count=(end-first+RowM-1)/RowM;}
  int total=0;
  const int before=prefix256(count,wave_totals,total);
  for(int b=0;b<count;++b)q.blocks[before+b]={tid,first+b*RowM,min(first+(b+1)*RowM,end),0};
  if(tid==0){q.state[NumBlocks]=total;q.state[NumL1]=total*q.n1;q.state[NumL2]=0;q.state[NumCombine]=0;q.state[HeadInput]=0;}
  int send_prefix=0;
  for(int base=0;base<l.num_experts;base+=256) {
    const int i=base+tid,dest=(i%l.num_ranks+p.view.rank)%l.num_ranks;
    const int expert=dest*l.num_experts_per_rank+i/l.num_ranks;
    int rows=0,start=0,first_rows=0,first_segments=0,segments=0;
    if(i<l.num_experts) {
      rows=region<const int>(p,l.counts)[p.view.rank*l.num_experts+expert];
      start=region<const int>(p,l.dst_rows)[expert];
      first_rows=min(rows,RowM-start%RowM);
      first_segments=(first_rows+DG_MOE_SEND_ROWS-1)/DG_MOE_SEND_ROWS;
      segments=first_segments+(rows-first_rows+DG_MOE_SEND_ROWS-1)/DG_MOE_SEND_ROWS;
    }
    int batch_total=0;const int offset=prefix256(segments,wave_totals,batch_total);
    for(int segment=0;segment<segments;++segment) {
      const int from=segment<first_segments?segment*DG_MOE_SEND_ROWS:first_rows+(segment-first_segments)*DG_MOE_SEND_ROWS;
      const int amount=min(DG_MOE_SEND_ROWS,(segment<first_segments?first_rows:rows)-from);
      q.push_blocks[send_prefix+offset+segment]={expert,from,start+from,amount};
    }
    send_prefix+=batch_total;
  }
  if(tid==0)q.state[NumPush]=send_prefix;
}

__device__ __forceinline__ void init(MegaMoePipelineArgs q) {
  for (int i = threadIdx.x; i < sizeof(q) / 4; i += blockDim.x)
    reinterpret_cast<uint32_t*>(q.context)[i] = reinterpret_cast<const uint32_t*>(&q)[i];
  for (int i = threadIdx.x; i < StateWords; i += blockDim.x) q.state[i] = 0;
  for (int i = threadIdx.x; i < q.max_blocks; i += blockDim.x) { q.l1_done[i] = q.l2_done[i] = q.sf_ready[i] = q.l1_issued[i] = 0; q.k_ready[i]=0; }
  if constexpr(!DG_MOE_FAST_INIT) {
    for (int i = threadIdx.x; i < q.max_blocks * q.n2; i += blockDim.x) q.job_ready[i] = 0;
    for (int i = threadIdx.x; i < q.max_blocks * 4; i += blockDim.x) q.combine_ready[i] = 0;
  }
  __syncthreads();
  if constexpr(DG_MOE_FAST_INIT){parallel_init_stage1(q);return;}
  if (threadIdx.x == 0) {
    int cursor = 0;
    for (int row = 0; row < q.shared_tokens; row += RowM)
      q.blocks[cursor++] = {0, row, min(row + RowM, q.shared_tokens), 1};
    const auto& l = q.moe.layout;
    const auto* ends = region<const int>(q.moe, l.psum_ends);
    for (int e = 0; e < l.num_experts_per_rank; ++e)
      for (int row = expert_start(ends, e, l.alignment); row < ends[e]; row += RowM)
        q.blocks[cursor++] = {e, row, min(row + RowM, ends[e]), 0};
    q.state[NumBlocks] = cursor;
    q.state[HeadInput]=(q.shared_tokens+RowM-1)/RowM;
    q.state[NumL1] = cursor * q.n1;
    q.state[NumL2] = cursor * q.n2;
    q.state[NumCombine] = DG_MOE_STAGE1_ONLY ? 0 : (cursor - (q.shared_tokens + RowM - 1) / RowM) * 4;
#if DG_MOE_SHARED_SPLIT
    const int shared_blocks=(q.shared_tokens+RowM-1)/RowM;
    for(int shard=0;shard<8;++shard)q.state[32+shard*32]=max(0,(shared_blocks*q.n1-shard+7)/8);
    for(int mb=0;mb<shared_blocks;++mb) {
      q.l1_done[mb]=q.l1_issued[mb]=q.n1;q.sf_ready[mb]=1;
      q.k_ready[mb]=q.n1==64?~uint64_t(0):(uint64_t(1)<<q.n1)-1;
      for(int col=0;col<q.n2;++col) {
        const int slot=(cursor-shared_blocks+mb)*q.n2+col;
        q.jobs[slot]=mb*q.n2+col;q.job_ready[slot]=1;
      }
    }
#endif
    if constexpr (DG_MOE_PUSH_BLOCKS) {
      int send=0;const auto& p=q.moe;const auto& l=p.layout;
      for(int i=0;i<l.num_experts;++i) {
        const int dest=(i%l.num_ranks+p.view.rank)%l.num_ranks;
        const int expert=dest*l.num_experts_per_rank+i/l.num_ranks;
        const int count=region<const int>(p,l.counts)[p.view.rank*l.num_experts+expert];
        const int first=region<const int>(p,l.dst_rows)[expert];
        for(int offset=0;offset<count;) {
          const int row=first+offset,rows=min(min(DG_MOE_SEND_ROWS,count-offset),RowM-row%RowM);
          q.push_blocks[send++]={expert,offset,row,rows};offset+=rows;
        }
      }
      q.state[NumPush]=send;
    }
  }
}

// Completion words are release-published. Consumers use coherent payload reads plus a
// compiler barrier after observing the word; this avoids invalidating immutable weight caches.
__device__ __forceinline__ void copy_coherent(uint8_t* dst, const uint8_t* src, int bytes, int lane) {
  for (int offset = lane * 16; offset < bytes; offset += 8192) {
    v4u v[8]{};
    // A single asm contains the asynchronous batch AND its wait. No compiler
    // copies or register reuse can be inserted while outputs are pending.
    const uint64_t address0 = reinterpret_cast<uint64_t>(src + offset);
    const uint64_t address4 = address0 + 4096;
    if (offset + 7 * 1024 < bytes) {
      asm volatile(
          "global_load_dwordx4 %0, %8, off sc1\n\t"
          "global_load_dwordx4 %1, %8, off offset:1024 sc1\n\t"
          "global_load_dwordx4 %2, %8, off offset:2048 sc1\n\t"
          "global_load_dwordx4 %3, %8, off offset:3072 sc1\n\t"
          "global_load_dwordx4 %4, %9, off sc1\n\t"
          "global_load_dwordx4 %5, %9, off offset:1024 sc1\n\t"
          "global_load_dwordx4 %6, %9, off offset:2048 sc1\n\t"
          "global_load_dwordx4 %7, %9, off offset:3072 sc1\n\t"
          "s_waitcnt vmcnt(0)"
          : "=&v"(v[0]), "=&v"(v[1]), "=&v"(v[2]), "=&v"(v[3]),
            "=&v"(v[4]), "=&v"(v[5]), "=&v"(v[6]), "=&v"(v[7])
          : "v"(address0), "v"(address4) : "memory");
    } else if (offset + 5 * 1024 < bytes) {
      asm volatile(
          "global_load_dwordx4 %0, %6, off sc1\n\t"
          "global_load_dwordx4 %1, %6, off offset:1024 sc1\n\t"
          "global_load_dwordx4 %2, %6, off offset:2048 sc1\n\t"
          "global_load_dwordx4 %3, %6, off offset:3072 sc1\n\t"
          "global_load_dwordx4 %4, %7, off sc1\n\t"
          "global_load_dwordx4 %5, %7, off offset:1024 sc1\n\t"
          "s_waitcnt vmcnt(0)"
          : "=&v"(v[0]), "=&v"(v[1]), "=&v"(v[2]), "=&v"(v[3]),
            "=&v"(v[4]), "=&v"(v[5])
          : "v"(address0), "v"(address4) : "memory");
      if (offset + 6 * 1024 < bytes) v[6] = coherent_load16(src + offset + 6 * 1024);
    } else {
#pragma unroll
      for (int i = 0; i < 8; ++i)
        if (offset + i * 1024 < bytes) v[i] = coherent_load16(src + offset + i * 1024);
    }
#pragma unroll
    for (int i = 0; i < 8; ++i)
      if (offset + i * 1024 < bytes) *reinterpret_cast<v4u*>(dst + offset + i * 1024) = v[i];
  }
}

__device__ __forceinline__ void prepare_row(const MegaMoePipelineArgs& q, const MegaMoeRowBlock& b, int mb);

__device__ __forceinline__ void worker_stamp(const MegaMoePipelineArgs& q, bool ending=false) {
#if DG_MOE_QUEUE_PROFILE
  if((threadIdx.x&63)==0) {
    const int at=blockIdx.x*4+threadIdx.x/64,stride=gridDim.x*4;
    const uint64_t id=(uint64_t(__smid())<<32)|__builtin_amdgcn_s_getreg((31u<<11)|4u);
    q.worker_ids[(ending?3:0)*stride+at]=id;
    q.worker_ids[(ending?2:1)*stride+at]=wall_clock64();
  }
#endif
}
__device__ __forceinline__ void comm_stamp(const MegaMoePipelineArgs& q,int index,int stage,int event) {
#if DG_MOE_QUEUE_PROFILE
  if((threadIdx.x&63)==0) {
    auto* r=q.comm_metrics+(int64_t(stage==2?5*q.max_blocks+index:stage?q.max_blocks+index:index)*4+threadIdx.x/64)*8;
    r[event]=wall_clock64();
    if(event==0){r[2]=blockIdx.x;r[3]=__smid();r[4]=stage;r[5]=index;r[6]=__builtin_amdgcn_s_getreg((31u<<11)|4u);r[7]=1;}
  }
  asm volatile("" ::: "memory");
#endif
}

// One producer owns a complete row block. Payload, scales and row metadata are
// published together, so consumers never wait on a different GEMM column to
// transpose scales. Source lists are immutable after the route/count barrier.
#if defined(DG_MOE_L1_K) && DG_MOE_L1_K>=1024 && DG_MOE_L1_K%1024==0 && DG_MOE_L1_K<=8192
// Compiler-managed buffer loads keep pending-register lifetimes visible to LLVM.
// Fetch several rows before any remote store, amortizing the store drain that a
// subsequent vmcnt(0) would otherwise impose once or twice per row.
template<int Bytes,int Batch,int Cache>
__device__ __forceinline__ void copy_row_batch(const uint8_t* const (&src)[Batch],
    uint8_t* const (&dst)[Batch],const bool (&valid)[Batch],int lane) {
  static_assert(Bytes%1024==0);
  constexpr int chunks=Bytes/1024;
  v4u values[Batch][chunks];
  static_for<0,Batch>([&](auto R){constexpr int r=decltype(R)::value;
    auto descriptor=make_rsrc(src[r],valid[r]?Bytes:0);
    static_for<0,chunks>([&](auto C){constexpr int c=decltype(C)::value;
      values[r][c]=__builtin_amdgcn_raw_buffer_load_b128(descriptor,lane*16,c*1024,Cache);
    });
  });
  asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
  static_for<0,Batch>([&](auto R){constexpr int r=decltype(R)::value;
    if(valid[r])static_for<0,chunks>([&](auto C){constexpr int c=decltype(C)::value;
      *reinterpret_cast<v4u*>(dst[r]+lane*16+c*1024)=values[r][c];
    });
  });
}

__device__ __forceinline__ void copy_output_batched(const MegaMoePipelineArgs& q,int ticket) {
  constexpr int Batch=2,Bytes=DG_MOE_L1_K*2;
  const int job=load_relaxed<Scope::device>(q.combine_jobs+ticket);
  const auto b=q.blocks[job/4];const auto& p=q.moe;const auto& l=p.layout;
  const int begin=b.m0+(job%4)*(RowM/4),end=min(b.end,begin+RowM/4);
  const int lane=threadIdx.x%64;
  for(int first=begin+(threadIdx.x/64)*Batch;first<end;first+=(blockDim.x/64)*Batch) {
    const uint8_t* src[Batch];uint8_t* dst[Batch];bool valid[Batch];
    static_for<0,Batch>([&](auto R){constexpr int r=decltype(R)::value;
      const int row=__builtin_amdgcn_readfirstlane(first+r);
      valid[r]=row<end;
      src[r]=reinterpret_cast<const uint8_t*>(p.l2_out)+int64_t(valid[r]?row:begin)*Bytes;
      dst[r]=nullptr;
      if(valid[r]) {
        const uint32_t source=load_relaxed<Scope::device>(region<const uint32_t>(p,l.l1_records)+int64_t(row)*l.record_ints+l.hidden/128+1);
        const int slot=source%l.num_topk,token=source/l.num_topk%l.num_max_tokens,rank=source/l.num_topk/l.num_max_tokens;
        dst[r]=p.view.map(region<uint8_t>(p,l.combine)+(int64_t(slot)*l.num_max_tokens+token)*Bytes,rank);
      }
    });
    copy_row_batch<Bytes,Batch,16>(src,dst,valid,lane);
  }
  __syncthreads();
}

__device__ __forceinline__ void push_segment_batched(const MegaMoePipelineArgs& q,int index) {
  constexpr int Batch=4,Bytes=DG_MOE_L1_K;
  const auto job=q.push_blocks[index];const auto& p=q.moe;const auto& l=p.layout;
  const int dest=job.expert/l.num_experts_per_rank,words=Bytes/128,lane=threadIdx.x%64;
  auto* out_base=p.view.map(region<uint8_t>(p,l.l1_acts),dest);
  auto* record_base=p.view.map(region<int>(p,l.l1_records),dest);
  for(int first=(threadIdx.x/64)*Batch;first<job.rows;first+=(blockDim.x/64)*Batch) {
    const uint8_t* src[Batch];uint8_t* dst[Batch];bool valid[Batch];v4u records[Batch];
    static_for<0,Batch>([&](auto R){constexpr int r=decltype(R)::value;
      const int i=first+r;valid[r]=i<job.rows;
      int entry=0;
      if(valid[r])entry=region<const int>(p,l.send_lists)[int64_t(job.expert)*l.num_max_tokens*l.num_topk+job.source_offset+i];
      entry=__builtin_amdgcn_readfirstlane(entry);
      const int token=entry/l.num_topk;
      src[r]=region<const uint8_t>(p,l.x)+int64_t(token)*Bytes;
      dst[r]=out_base+int64_t(job.dst_row+i)*Bytes;
      records[r]=v4u{};
      if(valid[r] && lane<l.record_ints/4) {
#pragma unroll
        for(int c=0;c<4;++c){const int w=lane*4+c;
          records[r][c]=w<words?uint32_t(region<const int>(p,l.x_sf)[int64_t(token)*words+w]):
            w==words?uint32_t(region<const int>(p,l.topk_weights)[entry]):
            w==words+1?uint32_t(p.view.rank*l.num_max_tokens*l.num_topk+entry):0;
        }
      }
    });
    copy_row_batch<Bytes,Batch,0>(src,dst,valid,lane);
    static_for<0,Batch>([&](auto R){constexpr int r=decltype(R)::value;
      if(valid[r] && lane<l.record_ints/4)
        *reinterpret_cast<v4u*>(record_base+int64_t(job.dst_row+first+r)*l.record_ints+lane*4)=records[r];
    });
  }
  fence_release<Scope::system>();__syncthreads();
  if(threadIdx.x==0)atomic_add_release<Scope::system>(p.view.map(input_flags(p)+job.dst_row/RowM,dest),job.rows);
  __syncthreads();
}

#endif

__device__ __forceinline__ void copy_output(const MegaMoePipelineArgs& q,int ticket) {
#if DG_MOE_COPY_BATCH && DG_MOE_L1_K>=1024 && DG_MOE_L1_K%1024==0 && DG_MOE_L1_K<=8192
  copy_output_batched(q,ticket);return;
#endif
  const int job=load_relaxed<Scope::device>(q.combine_jobs+ticket);
  const auto b=q.blocks[job/4];const auto& p=q.moe;const auto& l=p.layout;
  const int begin=b.m0+(job%4)*(RowM/4),end=min(b.end,begin+RowM/4);
  for(int row=begin+threadIdx.x/64;row<end;row+=blockDim.x/64) {
    const uint32_t source=load_relaxed<Scope::device>(region<const uint32_t>(p,l.l1_records)+int64_t(row)*l.record_ints+l.hidden/128+1);
    const int slot=source%l.num_topk,token=source/l.num_topk%l.num_max_tokens;
    const int rank=source/l.num_topk/l.num_max_tokens;
    auto* dst=p.view.map(region<uint8_t>(p,l.combine)+(int64_t(slot)*l.num_max_tokens+token)*l.hidden*2,rank);
    copy_coherent(dst,reinterpret_cast<const uint8_t*>(p.l2_out+int64_t(row)*l.hidden),l.hidden*2,threadIdx.x%64);
  }
  __syncthreads();
}

// Push a bounded segment whose destination lies in one receiver row block.
// A single release counter update publishes the segment after every lane's
// payload/metadata stores complete. This amortizes system-scope publication.
__device__ __forceinline__ void push_segment(const MegaMoePipelineArgs& q,int index) {
#if DG_MOE_COPY_BATCH && DG_MOE_L1_K>=1024 && DG_MOE_L1_K%1024==0 && DG_MOE_L1_K<=8192
  push_segment_batched(q,index);return;
#endif
  const auto job=q.push_blocks[index];const auto& p=q.moe;const auto& l=p.layout;
  const int dest=job.expert/l.num_experts_per_rank,words=l.hidden/128;
  for(int i=threadIdx.x/64;i<job.rows;i+=blockDim.x/64) {
    const int entry=region<const int>(p,l.send_lists)[int64_t(job.expert)*l.num_max_tokens*l.num_topk+job.source_offset+i];
    const int token=entry/l.num_topk,row=job.dst_row+i;
    const auto* input=region<const uint8_t>(p,l.x)+int64_t(token)*l.hidden;
    auto* out=p.view.map(region<uint8_t>(p,l.l1_acts)+int64_t(row)*l.hidden,dest);
    copy_coherent(out,input,l.hidden,threadIdx.x%64);
    auto* record=p.view.map(region<int>(p,l.l1_records)+int64_t(row)*l.record_ints,dest);
    for(int pack=threadIdx.x%64;pack<l.record_ints/4;pack+=64) {
      v4u value{};
#pragma unroll
      for(int c=0;c<4;++c) {
        const int w=pack*4+c;
        value[c]=w<words?uint32_t(region<const int>(p,l.x_sf)[int64_t(token)*words+w]):
          w==words?uint32_t(region<const int>(p,l.topk_weights)[entry]):
          w==words+1?uint32_t(p.view.rank*l.num_max_tokens*l.num_topk+entry):0;
      }
      *reinterpret_cast<v4u*>(record+pack*4)=value;
    }
  }
  fence_release<Scope::system>();__syncthreads();
  if(threadIdx.x==0)atomic_add_release<Scope::system>(
    p.view.map(input_flags(p)+job.dst_row/RowM,dest),job.rows);
  __syncthreads();
}

__device__ __forceinline__ void dispatch_push(const MegaMoePipelineArgs& q) {
  // action/item, pending output ticket, pending local input block.
  __shared__ int decision[4];
  if(threadIdx.x==0)decision[2]=decision[3]=-1;
  __syncthreads();
  for(;;) {
    if(threadIdx.x==0) {
      int& output=decision[2];int& input=decision[3];
      if(output==-1){output=atomic_add<Scope::device>(q.state+HeadCombine,1);if(output>=q.state[NumCombine])output=-2;}
      if(input==-1){input=atomic_add<Scope::device>(q.state+HeadInput,1);if(input>=q.state[NumBlocks])input=-2;}
      bool arrived=false;
      if(input>=0){const auto b=q.blocks[input];arrived=load_relaxed<Scope::system>(input_flags(q.moe)+b.m0/RowM)==b.end-b.m0;}
      if(arrived){decision[0]=0;decision[1]=input;input=-1;}
      else if(output>=0 && load_relaxed<Scope::device>(q.combine_ready+output)) {
        decision[0]=1;decision[1]=output;output=-1;
      } else {
        int send=-1;
        if(load_relaxed<Scope::device>(q.state+HeadPush)<q.state[NumPush])send=atomic_add<Scope::device>(q.state+HeadPush,1);
        if(send>=0 && send<q.state[NumPush]){decision[0]=2;decision[1]=send;}
        else decision[0]=(input==-2 && output==-2)?4:3;
      }
    }
    __syncthreads();const int action=decision[0],item=decision[1];
    if(action==4)break;
    if(action==0){comm_stamp(q,item,0,0);prepare_row(q,q.blocks[item],item);comm_stamp(q,item,0,1);}
    else if(action==1){const int id=load_relaxed<Scope::device>(q.combine_jobs+item);comm_stamp(q,id,1,0);copy_output(q,item);comm_stamp(q,id,1,1);}
    else if(action==2){comm_stamp(q,item,2,0);push_segment(q,item);comm_stamp(q,item,2,1);}
    else __builtin_amdgcn_s_sleep(8);
    __syncthreads();
  }
  fence_release<Scope::system>();
}

// Every actor has already completed its system-scope payload fence. The last
// actor publishes output completion to peers while local shared-L2 can continue.
__device__ __forceinline__ void signal_outputs(const MegaMoePipelineArgs& q) {
#if DG_MOE_EARLY_READY
  __shared__ int last;
  if(threadIdx.x==0)last=atomic_add_acq_rel<Scope::device>(q.state+CommDone,1)==q.dispatch_blocks-1;
  __syncthreads();
  if(last && threadIdx.x<q.moe.view.world_size) {
    const auto& p=q.moe;const int offset=(p.layout.max_rows+127)/128+p.view.rank;
    store_release<Scope::system>(p.view.map(region<int>(p,p.layout.output_rows)+offset,threadIdx.x),1);
  }
  fence_release<Scope::system>();__syncthreads();
#endif
}
__device__ __forceinline__ void wait_outputs(const MegaMoePipelineArgs& q) {
  const auto& p=q.moe;
  if(threadIdx.x<p.view.world_size) {
    const auto* flag=region<const int>(p,p.layout.output_rows)+(p.layout.max_rows+127)/128+threadIdx.x;
    const auto begin=clock_ns();
    while(!load_acquire<Scope::system>(flag)) {
      __builtin_amdgcn_s_sleep(8);
      if(clock_ns()-begin>60000000000ull){printf("output completion timeout rank=%d peer=%d\n",p.view.rank,threadIdx.x);trap();}
    }
  }
  __syncthreads();
}

// Payload cache lines are single-assignment within a call. Complete every
// resident worker's epoch invalidation before any local producer/consumer runs.
// Tile/K boundaries are cache-line aligned; no speculative next-tile A load is used.
__device__ __forceinline__ void cache_epoch(const MegaMoePipelineArgs& q) {
#if DG_MOE_CACHE_A == 2
  fence_acquire<Scope::system>();__syncthreads();
  if(threadIdx.x==0)atomic_add_acq_rel<Scope::device>(q.state+CacheEpoch,1);
  __syncthreads();
  if(threadIdx.x==0)while(load_relaxed<Scope::device>(q.state+CacheEpoch)<int(gridDim.x))__builtin_amdgcn_s_sleep(4);
  __syncthreads();asm volatile("" ::: "memory");
#endif
}

__device__ __forceinline__ bool is_dispatch(const MegaMoePipelineArgs& q) {
  return blockIdx.x<q.dispatch_blocks;
}

__device__ __forceinline__ void dispatch(const MegaMoePipelineArgs& q) {
  worker_stamp(q);cache_epoch(q);dispatch_push(q);signal_outputs(q);worker_stamp(q,true);
}

// The single thread scheduler never claims an unavailable L1 task. Ready L2 tasks have
// priority, so workgroups can consume a finished row block while others still execute L1.
__device__ __forceinline__ int pick(const MegaMoePipelineArgs& q, int& stage, int& pending1, int& pending2) {
  const auto start = clock_ns();
  const int shard = blockIdx.x % 8;
  auto input_ready = [&](int index) {
    const auto b = q.blocks[index / q.n1];
    const auto& p = q.moe;
    if constexpr (DG_MOE_BLOCK_PIPELINE) return b.shared || load_relaxed<Scope::device>(q.sf_ready+index/q.n1);
    return b.shared || load_relaxed<Scope::system>(region<const int>(p, p.layout.output_rows) +
                        b.expert / q.input_group) >= p.view.world_size * q.dispatch_blocks;
  };
  for (;;) {
    if (load_relaxed<Scope::device>(q.state + DoneL2) == q.state[NumL2]) return -1;
    if (pending1 == -1) {
      const int ticket = atomic_add<Scope::device>(q.state + 32 + shard * 32, 1) * 8 + shard;
      pending1 = ticket < q.state[NumL1] ? ticket : -2;
    }
    if (pending2 == -1) {
      const int ticket = atomic_add<Scope::device>(q.state + 288 + shard * 32, 1) * 8 + shard;
      pending2 = ticket < q.state[NumL2] ? ticket : -2;
    }
    // Prioritize the task that transposes scales: other columns of its row block wait for it.
    if (pending1 >= 0 && pending1 % q.n1 == 0 && input_ready(pending1)) {
      const int index = pending1; pending1 = -1; stage = 0; return index;
    }
    if (pending2 >= 0 && load_relaxed<Scope::device>(q.job_ready + pending2)) {
      const int index = load_relaxed<Scope::device>(q.jobs + pending2);
      if(!DG_MOE_STREAM_K || __builtin_popcountll(load_relaxed<Scope::device>(q.k_ready+index/q.n2))>=q.min_l1_ready) {
        pending2 = -1; stage = 1; return index;
      }
    }
    if (pending1 >= 0 && input_ready(pending1)) {
      const int index = pending1; pending1 = -1; stage = 0; return index;
    }
    __builtin_amdgcn_s_sleep(8);
    if (clock_ns() - start > 60000000000ull) {
      printf("MoE queue pick timeout rank %d done %d/%d\n", q.moe.view.rank, q.state[DoneL2], q.state[NumL2]); trap();
    }
  }
}

__device__ __forceinline__ void prepare_row(const MegaMoePipelineArgs& q, const MegaMoeRowBlock& b, int mb) {
  const auto& p = q.moe;
  const auto& l = p.layout;
  const int words = l.hidden / 128, rows = b.end - b.m0;
  const auto* src = region<const int>(p, b.shared ? l.x_sf : l.l1_records);
  auto* dst = region<int>(p, b.shared ? l.shared_l1_sf : l.l1_sf);
  const int src_pitch = b.shared ? words : l.record_ints, dst_pitch = b.shared ? l.shared_sf_pitch : l.sf_pitch;
  for (int row = b.m0 + threadIdx.x % 128; row < b.end; row += 128) {
    for (int pack = threadIdx.x / 128; pack < (words + 3) / 4; pack += 2) {
      const auto* source = src + int64_t(row) * src_pitch + pack * 4;
      v4u v{};
      if (src_pitch % 4 == 0 && pack * 4 + 4 <= words)
        v = coherent_load16(source);
      else
        for (int c = 0; c < 4 && pack * 4 + c < words; ++c)
          v[c] = load_relaxed<Scope::system>(source + c);
#pragma unroll
      for (int c = 0; c < 4; ++c)
        if (pack * 4 + c < words)
          store_relaxed<Scope::device>(dst + int64_t(pack * 4 + c) * dst_pitch + row, int(v[c]));
    }
  }
#if DG_MOE_EARLY_POINTERS
  for(int row=b.m0+threadIdx.x;row<b.end;row+=blockDim.x) {
    const uint32_t source=load_relaxed<Scope::system>(region<const uint32_t>(p,l.l1_records)+int64_t(row)*l.record_ints+words+1);
    const int slot=source%l.num_topk,token=source/l.num_topk%l.num_max_tokens,rank=source/l.num_topk/l.num_max_tokens;
    region<int64_t>(p,l.output_rows)[row]=reinterpret_cast<int64_t>(p.view.map(region<uint16_t>(p,l.combine)+(int64_t(slot)*l.num_max_tokens+token)*l.hidden,rank));
  }
#endif
  fence_release<Scope::device>();
  __syncthreads();
  if (threadIdx.x == 0) store_release<Scope::device>(q.sf_ready + mb, 1);
  __syncthreads();
}

__device__ __forceinline__ int reserve_l2(const MegaMoePipelineArgs& q,int mb) {
#if DG_MOE_SHARED_TAIL
  const int shared=(q.shared_tokens+RowM-1)/RowM;
  if(mb<shared)return (q.state[NumBlocks]-shared+mb)*q.n2;
#endif
  return atomic_add<Scope::device>(q.state+TailL2,q.n2);
}

template<bool Shared,bool Second>
__device__ __forceinline__ GemmArgs stage_args(const MegaMoePipelineArgs& q) {
#if DG_MOE_RELOAD_ARGS
  uint64_t address=uint64_t(__builtin_amdgcn_kernarg_segment_ptr());
  asm volatile("s_mov_b64 %0, %0" : "+s"(address) :: "memory");
  const auto* params=reinterpret_cast<const __attribute__((address_space(4))) MegaMoePipelineArgs*>(address);
  if constexpr(Shared && Second)return params->shared_l2;
  else if constexpr(Shared)return params->shared_l1;
  else if constexpr(Second)return params->l2;
  else return params->l1;
#else
  if constexpr(Shared && Second)return q.shared_l2;
  else if constexpr(Shared)return q.shared_l1;
  else if constexpr(Second)return q.l2;
  else return q.l1;
#endif
}

template <bool Shared, bool Second>
__device__ __forceinline__ void tile(GemmArgs a,const MegaMoeRowBlock& b,int n0,uint64_t* profile,const uint64_t* k_ready=nullptr) {
  if constexpr(Shared){if constexpr(Second)NV_CONST_S2(a);else NV_CONST_S1(a);}
  else {if constexpr(Second)NV_CONST_L2(a);else NV_CONST_L1(a);}
  if constexpr(Second) pipeline_native_tile<DG_MOE_L2_K,DG_MOE_L2_N,false,!Shared>(a,b.expert,b.m0,b.end,n0,profile,k_ready);
  else pipeline_native_tile<DG_MOE_L1_K,DG_MOE_L1_N,true,!Shared>(a,b.expert,b.m0,b.end,n0,profile,k_ready);
  if constexpr(!Second) {
    extern __shared__ __attribute__((aligned(1024))) uint8_t smem[];
    const int tid=threadIdx.x,r=tid/8;
#pragma unroll 1
    for(int round=0;round<RowM/32;++round) {
      const int extra_rows=16*round+(r/16)*(RowM/2-16);
      pipeline_swiglu_epilogue<false,RowM/2,true>(a,smem+extra_rows*528,528,b.m0,b.end,n0,round,tid);
    }

    asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)\n\ts_barrier" ::: "memory");
  }
}

#ifndef DG_MOE_QUEUE_PROFILE
#define DG_MOE_QUEUE_PROFILE 0
#endif
// Unique task/wave slots avoid counters and keep no timing state live across GEMM.
__device__ __forceinline__ void stamp(const MegaMoePipelineArgs& q, int task, bool second, int event) {
#if DG_MOE_QUEUE_PROFILE
  if ((threadIdx.x & 63) == 0) {
    const int index = (second ? q.max_blocks * q.n1 : 0) + task;
    auto* record = q.metrics + (int64_t(index) * 4 + threadIdx.x / 64) * 12;
    record[event] = wall_clock64();
    if (event == 0) { record[4] = blockIdx.x; record[5] = second; record[6] = task; record[7] = 1; }
  }
  asm volatile("" ::: "memory");
#endif
}

// Dedicated producer/consumer compute roles: L2 cannot occupy L1 resources.
// This also lets the compiler keep only one GEMM stage's arguments live.
template<int Stage>
__device__ __forceinline__ int pick_stage(const MegaMoePipelineArgs& q,int& pending) {
  constexpr bool second=Stage==1;const int shard=blockIdx.x%8;
  const int n1=DG_MOE_L1_LOCAL_ONLY?DG_MOE_L1_N/256:q.n1;
  if(pending<0) {
    const int ticket=atomic_add<Scope::device>(q.state+(second?288:32)+shard*32,1)*8+shard;
    if(ticket>=q.state[second?NumL2:NumL1])return -1;
    pending=ticket;
  }
  const auto begin=clock_ns();
  for(;;) {
    if constexpr(second) {
      if(load_relaxed<Scope::device>(q.job_ready+pending)) {
        const int task=load_relaxed<Scope::device>(q.jobs+pending);
        if(!DG_MOE_STREAM_K || __builtin_popcountll(load_relaxed<Scope::device>(q.k_ready+task/q.n2))>=q.min_l1_ready) {pending=-1;return task;}
      }
    } else {
      const auto b=q.blocks[pending/n1];bool ready=b.shared;
      if constexpr(DG_MOE_BLOCK_PIPELINE)ready=ready||load_relaxed<Scope::device>(q.sf_ready+pending/n1);
      else ready=ready||load_relaxed<Scope::system>(region<const int>(q.moe,q.moe.layout.output_rows)+b.expert/q.input_group)>=q.moe.view.world_size*q.dispatch_blocks;
      if(ready){const int task=pending;pending=-1;return task;}
    }
    __builtin_amdgcn_s_sleep(8);
    if(clock_ns()-begin>60000000000ull){printf("pipeline stage wait timeout rank=%d stage=%d ticket=%d\n",q.moe.view.rank,Stage,pending);trap();}
  }
}

template<int ForcedStage=-1>
__device__ __forceinline__ void compute(const MegaMoePipelineArgs& q) {
  worker_stamp(q);
  cache_epoch(q);
  __shared__ int job[6];
  if (threadIdx.x == 0) job[3] = job[4] = job[5] = -1;
  __syncthreads();
  for (;;) {
    if (threadIdx.x == 0) {
      if constexpr(ForcedStage<0)job[0]=pick(q,job[1],job[3],job[4]);
      else job[0]=pick_stage<ForcedStage>(q,job[3]);
    }
    __syncthreads();
    const int task = __builtin_amdgcn_readfirstlane(job[0]);
    if (task < 0) break;
    const bool second = ForcedStage<0?__builtin_amdgcn_readfirstlane(job[1]):ForcedStage==1;
    const int columns = DG_MOE_L1_LOCAL_ONLY ? DG_MOE_L1_N/256 : (second ? q.n2 : q.n1), mb = task / columns, col = task % columns;
    const auto b = q.blocks[mb];
    uint64_t* profile = nullptr;
#if DG_MOE_QUEUE_PROFILE
    profile = q.metrics + (int64_t(second ? q.max_blocks * q.n1 : 0) + task) * 4 * 12;
#endif
    stamp(q, task, second, 0);
    asm volatile("" ::: "memory");
    if (!second) {
      if constexpr(!DG_MOE_L1_LOCAL_ONLY) {
        if (col == 0 && (!DG_MOE_COMM_SF || b.shared || q.dispatch_blocks==0)) prepare_row(q, b, mb);
        else wait_local(q.sf_ready + mb, 1);
      }
      // In the local-only Stage1 variant pick_stage already observed sf_ready;
      // all waves rendezvoused after the choice. Tile A/SF loads are coherent.

#if DG_MOE_CACHE_A == 1
      fence_acquire<Scope::system>();
      __syncthreads();
#endif
      // Admit L2 only when every producer task is actually committed to a
      // workgroup. No required L1 tile may remain held in a pending ticket.
      if constexpr (DG_MOE_STREAM_K) {
        if(threadIdx.x==0) {
          const int issued=atomic_add<Scope::device>(q.l1_issued+mb,1);
          job[2]=issued==q.n1-1?reserve_l2(q,mb):-1;
        }
        __syncthreads();
        if(job[2]>=0)for(int j=threadIdx.x;j<q.n2;j+=blockDim.x) {
          const int slot=job[2]+j;q.jobs[slot]=mb*q.n2+j;
          store_release<Scope::device>(q.job_ready+slot,1);
        }
        __syncthreads();
      }
      stamp(q, task, second, 1);
      if constexpr(DG_MOE_SHARED_SPLIT) tile<false,false>(stage_args<false,false>(q),b,col*256,profile);
      else if (b.shared) tile<true, false>(stage_args<true,false>(q), b, col * 256, profile);
      else tile<false, false>(stage_args<false,false>(q), b, col * 256, profile);
      stamp(q, task, second, 2);
#if DG_MOE_L1_ONLY_PUBLISH
      stamp(q, task, second, 3);
      __syncthreads();
      continue;
#endif
      asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
      __syncthreads();
      if (threadIdx.x == 0) {
        const int old = atomic_add_release<Scope::device>(q.l1_done + mb, 1);
        if constexpr (DG_MOE_STREAM_K) atomic_add_release<Scope::device>(q.k_ready+mb,uint64_t(1)<<col);
        job[2] = !DG_MOE_STAGE1_ONLY && !DG_MOE_STREAM_K && old == q.n1 - 1 ? reserve_l2(q,mb) : -1;
      }
      __syncthreads();
      if (job[2] >= 0)
        for (int j = threadIdx.x; j < q.n2; j += blockDim.x) {
          const int slot = job[2] + j;
          q.jobs[slot] = mb * q.n2 + j;
          store_release<Scope::device>(q.job_ready + slot, 1);
        }
    } else {
#if DG_MOE_CACHE_A == 1
      fence_acquire<Scope::system>();
      __syncthreads();
#endif
      stamp(q, task, second, 1);
      if (b.shared) tile<true, true>(stage_args<true,true>(q), b, col * 256, profile, DG_MOE_STREAM_K?q.k_ready+mb:nullptr);
      else tile<false, true>(stage_args<false,true>(q), b, col * 256, profile, DG_MOE_STREAM_K?q.k_ready+mb:nullptr);
      stamp(q, task, second, 2);
      asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
      __syncthreads();
      if (threadIdx.x == 0) {
        const int old = atomic_add_release<Scope::device>(q.l2_done + mb, 1);
        job[2] = !b.shared && old == q.n2 - 1 ? atomic_add<Scope::device>(q.state + TailCombine, 4) : -1;
        atomic_add_release<Scope::device>(q.state + DoneL2, 1);
      }
      __syncthreads();
      if (job[2] >= 0 && threadIdx.x < 4) {
        const int slot = job[2] + threadIdx.x;
        q.combine_jobs[slot] = mb * 4 + threadIdx.x;
        store_release<Scope::device>(q.combine_ready + slot, 1);
      }
    }
    stamp(q, task, second, 3);
    __syncthreads();
  }
  // Only local L1 outputs are produced here. The following kernel on the same
  // stream consumes them after cooperative-kernel completion. Sender publication
  // fences remain in push_segment/dispatch and are never removed by this flag.
  if constexpr(!DG_MOE_L1_LOCAL_ONLY)fence_release<Scope::system>();
  worker_stamp(q,true);
}
}
