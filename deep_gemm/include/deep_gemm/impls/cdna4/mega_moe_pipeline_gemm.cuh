#pragma once
#include <deep_gemm/impls/cdna4/device_common.h>
#include <deep_gemm/impls/cdna4/gemm_args.h>
using v16f = float __attribute__((ext_vector_type(16)));
#ifndef DG_MOE_NATIVE_ALIAS_L2
#define DG_MOE_NATIVE_ALIAS_L2 0
#endif
#define G32_STORE_POLICY 0
#define G32_DIRECT 0
#define G32_WIDE_N 0
#define G32_INTERTILE 0
#define G32_PAIR_B 0
#define G32_PROFILE 0
#define G32_STAGES 2
#define G32_PERSISTENT 0
// Narrow hot path: constant shapes, 32x32x64 MFMA, four waves, 256x256 tile.
// Only routed FP8xFP4; shared experts keep the generic FP8xFP8 implementation.
template<int K,int N,bool First,bool PackedB=true>
__device__ __forceinline__ void pipeline_native_tile(deep_gemm::cdna4::GemmArgs p,int expert,int m0,int end,int n0,uint64_t* profile,const uint64_t* k_ready) {
 constexpr int KT=K/128,S=2;
 constexpr int BM=G32_WIDE_N?128:256,BN=G32_WIDE_N?512:256;
 constexpr int AL=BM/32,BL=BN/(PackedB?64:32),DMAS=AL+BL+1+BN/256;
 constexpr int BROW=PackedB?64:128,BSTRIDE=PackedB?K/2:K;
 // SFA reserves256 rows: spare waves write distinct dummy rows, not overlapping stores.
 constexpr int AO=0,BO=BM*128,SAO=BO+BN*BROW,SBO=SAO+256*4,STEP=SBO+BN*4;
 constexpr int EPI=S*STEP,PITCH=2*BN+16,EROWS=G32_WIDE_N?32:64;
 static_assert(!G32_DIRECT || G32_WIDE_N);
 static_assert(K%128==0 && K>=256 && N%BN==0 && S==2);
 static_assert((EPI>BM*PITCH?EPI:BM*PITCH)<=160*1024);
 extern __shared__ __attribute__((aligned(1024))) uint8_t sm[];
 const int tid=threadIdx.x,lane=tid%64,wave=__builtin_amdgcn_readfirstlane(tid/64);
 const int wm=G32_WIDE_N?0:wave/2,wn=G32_WIDE_N?wave:wave%2,r=lane%32,g=lane/32;
 const int ar=8*wave+lane/8,br=PackedB?16*wave+lane/4:8*wave+lane/8;
 const int bkey_dma=PackedB?((br>>2)&3):((br>>1)&7);
 const int brow=G32_PAIR_B?((r&3)+((r>>3)*4)+(r&4)*4):r;
 const int av=ar*K+((lane%8)^((ar>>1)&7))*16;
 const int bv=br*BSTRIDE+((lane%(PackedB?4:8))^bkey_dma)*16;
 const int sv=(wave*64+lane)*4,sa_v=((wave%(BM/64))*64+lane)*4;
 const int aoff=AO+(wm*128+r)*128,boff=BO+(wn*128+brow)*BROW;
 const int akey=(r>>1)&7,bkey=PackedB?((r>>2)&3):((r>>1)&7);
 const int aso=SAO+(wm*128+r)*4,bso=SBO+(wn*128+brow)*4;
 auto stamp=[&](int event) {
#if DG_MOE_QUEUE_PROFILE
  if(lane==0)profile[wave*12+event]=wall_clock64();
  asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
#endif
 };
 constexpr int tile_phase=0;
 uint64_t ready_cache=0;
  auto ra=make_rsrc(p.A+int64_t(m0)*K,min(end-m0,BM)*K);
  auto rb=make_rsrc(p.B+int64_t(expert)*p.b_group_stride+int64_t(n0)*BSTRIDE,BN*BSTRIDE);
  auto rsa=make_rsrc(p.sfa+m0,uint32_t((uint64_t(K/128-1)*p.lda_sf+min(end-m0,BM))*4));
  auto rsb=make_rsrc(p.sfb+int64_t(expert)*p.sfb_group_stride+n0,uint32_t((uint64_t(K/128-1)*p.ldb_sf+BN)*4));
  auto dma_one=[&](auto J,int stage,int k) {
   constexpr int j=decltype(J)::value;
   uint8_t* s=sm+stage*STEP;
#if DG_MOE_STREAM_K
   if constexpr(!First && j==0) {
    if(k_ready)while(!(ready_cache&(uint64_t(1)<<k))) {
     ready_cache=__hip_atomic_load(k_ready,__ATOMIC_RELAXED,__HIP_MEMORY_SCOPE_AGENT);
     if(!(ready_cache&(uint64_t(1)<<k)))__builtin_amdgcn_s_sleep(4);
    }
   }
#endif
   if constexpr (j<AL)
    __builtin_amdgcn_raw_ptr_buffer_load_lds(ra,(lds_ptr_t)(s+AO+(j*4+wave)*1024),16,av,j*32*K+k*128,0,DG_MOE_CACHE_A?0:16);
   else if constexpr(j<AL+BL)
    __builtin_amdgcn_raw_ptr_buffer_load_lds(rb,(lds_ptr_t)(s+BO+((j-AL)*4+wave)*1024),16,bv,(j-AL)*(PackedB?64:32)*BSTRIDE+k*BROW,0,0);
   else if constexpr(j==AL+BL)
    __builtin_amdgcn_raw_ptr_buffer_load_lds(rsa,(lds_ptr_t)(s+SAO+wave*256),4,sa_v,uint32_t(k)*uint32_t(p.lda_sf)*4u,0,DG_MOE_CACHE_A?0:16);
   else
    __builtin_amdgcn_raw_ptr_buffer_load_lds(rsb,(lds_ptr_t)(s+SBO+(wave+(j-AL-BL-1)*4)*256),4,sv,uint32_t(k)*uint32_t(p.ldb_sf)*4u+(j-AL-BL-1)*1024u,0,0);
  };
  auto dma=[&](int stage,int k){static_for<0,DMAS>([&](auto J){dma_one(J,stage,k);});};
  v16f acc[4][4];
  v8i af[2][4],bf[2][4];int ac[2][4],bc[2][4];
  auto read_frag=[&](auto Bank,auto F,int half,int stage) {
   constexpr int bank=decltype(Bank)::value,f=decltype(F)::value;
   const uint8_t* st=sm+stage*STEP;
   if constexpr(f>=1 && f<=4) {
    constexpr int i=f-1;
    const v4i lo=*reinterpret_cast<const v4i*>(st+aoff+i*32*128+((half*4+g)^akey)*16);
    const v4i hi=*reinterpret_cast<const v4i*>(st+aoff+i*32*128+((half*4+g+2)^akey)*16);
    af[bank][i]=v8i{lo[0],lo[1],lo[2],lo[3],hi[0],hi[1],hi[2],hi[3]};
    ac[bank][i]=st[aso+i*32*4+half*2+g];
   } else {
    constexpr int i=f==0?0:f-4;
    if constexpr(PackedB) {
      const v4i b=*reinterpret_cast<const v4i*>(st+boff+i*32*BROW+((half*2+g)^bkey)*16);
      bf[bank][i]=v8i{b[0],b[1],b[2],b[3],0,0,0,0};
    } else {
      const v4i lo=*reinterpret_cast<const v4i*>(st+boff+i*32*BROW+((half*4+g)^bkey)*16);
      const v4i hi=*reinterpret_cast<const v4i*>(st+boff+i*32*BROW+((half*4+g+2)^bkey)*16);
      bf[bank][i]=v8i{lo[0],lo[1],lo[2],lo[3],hi[0],hi[1],hi[2],hi[3]};
    }
    bc[bank][i]=st[bso+i*32*4+half*2+g];
   }
  };
  auto step=[&](int k,auto first,auto last) {
   const int stage=(k+tile_phase)&1,next=1-stage,loadk=k+2<KT?k+2:k+2-KT;
   static_for<0,32>([&](auto T){constexpr int t=decltype(T)::value,half=t/16,i=(t%16)/4,j=t%4;
    acc[i][j]=__builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(bf[half][j],af[half][i],decltype(first)::value && half==0?v16f{}:acc[i][j],PackedB?4:0,0,0,bc[half][j],0,ac[half][i]);
    if constexpr(t>=8 && t<16) read_frag(std::integral_constant<int,1>{},std::integral_constant<int,t-8>{},1,stage);
    if constexpr(!decltype(last)::value || G32_INTERTILE) {
     if constexpr(t==19) asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory");
     static_for<0,DMAS>([&](auto J){constexpr int j=decltype(J)::value;
      if constexpr(t==20+j*12/DMAS) dma_one(J,stage,loadk);
     });
     if constexpr(t==24) {
      __builtin_amdgcn_s_waitcnt(waitcnt_vm((5*DMAS+11)/12));
      asm volatile("s_barrier" ::: "memory");
     }
     if constexpr(t>=24 && !decltype(last)::value) read_frag(std::integral_constant<int,0>{},std::integral_constant<int,t-24>{},0,next);
    }
    __builtin_amdgcn_sched_barrier(0);
   });
  };
  stamp(8);
  if(true) {
   static_for<0,2>([&](auto I){dma(decltype(I)::value,decltype(I)::value);});
   __builtin_amdgcn_s_waitcnt(waitcnt_vm(DMAS));
   asm volatile("s_barrier" ::: "memory");
  }
  static_for<0,8>([&](auto F){read_frag(std::integral_constant<int,0>{},F,0,tile_phase);});
  stamp(11);
  step(0,std::true_type{},std::false_type{});
#pragma unroll 1
  for(int k=1;k<KT-1;++k)step(k,std::false_type{},std::false_type{});
  step(KT-1,std::false_type{},std::true_type{});
  if constexpr(G32_INTERTILE) asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory");
  else asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)\n\ts_barrier" ::: "memory");
  stamp(9);
  if constexpr(First || !PackedB || DG_MOE_NATIVE_ALIAS_L2) {
  {
   // Alias the fully drained operand LDS with the complete BF16 output tile.
   // Accumulators die before the caller runs SwiGLU/quantization.
   static_for<0,4>([&](auto I){constexpr int i=decltype(I)::value;
    static_for<0,4>([&](auto J){constexpr int j=decltype(J)::value;
     static_for<0,4>([&](auto Q){constexpr int q=decltype(Q)::value;
      const int dst=(wm*128+i*32+r)*528+(wn*128+j*32+g*4+q*8)*2;
      *reinterpret_cast<v2u*>(sm+dst)=v2u{pack_bf16x2(acc[i][j][q*4],acc[i][j][q*4+1]),pack_bf16x2(acc[i][j][q*4+2],acc[i][j][q*4+3])};
      __builtin_amdgcn_sched_barrier(0);
     });
    });
   });
   asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory");
  }
  if constexpr(!First) {
    uint32_t etid=tid;asm volatile("v_mov_b32 %0, %0" : "+v"(etid) :: "memory");
    const int row0=etid/32,col=(etid%32)*8;
    auto out=make_rsrc(reinterpret_cast<uint16_t*>(p.D)+int64_t(m0)*N+n0,
                       uint32_t((min(end-m0,BM)-1)*N*2+BN*2));
    const uint32_t voff=row0*N*2+col*2;
    static_for<0,32>([&](auto Q){constexpr int q=decltype(Q)::value;
      const v4u value=*reinterpret_cast<const v4u*>(sm+(row0+q*8)*528+col*2);
      buffer_store16<0>(value,out,voff,q*8*N*2);
    });
    asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)\n\ts_barrier" ::: "memory");
  }
  } else {

  static_for<0,4>([&](auto I){constexpr int i=decltype(I)::value;
   static_for<0,4>([&](auto J){constexpr int j=decltype(J)::value;
    if constexpr(G32_PAIR_B) {
     static_for<0,2>([&](auto Q){constexpr int q=decltype(Q)::value;
      const int dst=EPI+(wm*32+r)*PITCH+(wn*128+j*32+g*16+q*8)*2;
      *reinterpret_cast<v4u*>(sm+dst)=v4u{
       pack_bf16x2(acc[i][j][q*8],acc[i][j][q*8+1]),
       pack_bf16x2(acc[i][j][q*8+2],acc[i][j][q*8+3]),
       pack_bf16x2(acc[i][j][q*8+4],acc[i][j][q*8+5]),
       pack_bf16x2(acc[i][j][q*8+6],acc[i][j][q*8+7])};
      __builtin_amdgcn_sched_barrier(0);
     });
    } else static_for<0,4>([&](auto Q){constexpr int q=decltype(Q)::value;
     const int dst=EPI+(wm*32+r)*PITCH+(wn*128+j*32+g*4+q*8)*2;
     *reinterpret_cast<v2u*>(sm+dst)=v2u{pack_bf16x2(acc[i][j][q*4],acc[i][j][q*4+1]),
                                          pack_bf16x2(acc[i][j][q*4+2],acc[i][j][q*4+3])};
     __builtin_amdgcn_sched_barrier(0);
    });
   });
   asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory");
   static_for<0,8>([&](auto Q){constexpr int q=decltype(Q)::value;
    const int lr=G32_WIDE_N?wave+q*4:tid/32+q*8;
    const int row=G32_WIDE_N?m0+i*32+lr:m0+(lr/32)*128+i*32+lr%32;
    const int c=G32_WIDE_N?lane:tid%32,col=n0+c*8;
    const v4u v=*reinterpret_cast<const v4u*>(sm+EPI+lr*PITCH+c*16);
    if(row<end) {
     *reinterpret_cast<v4u*>(reinterpret_cast<uint16_t*>(p.D)+int64_t(row)*N+col)=v;
    }
   });
   asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory");
  });
  if constexpr(G32_DIRECT) asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory");
  else asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)\n\ts_barrier" ::: "memory");
  
  }
  stamp(10);
}

#undef G32_STORE_POLICY
#undef G32_DIRECT
#undef G32_WIDE_N
#undef G32_INTERTILE
#undef G32_PAIR_B
#undef G32_PROFILE
#undef G32_STAGES
#undef G32_PERSISTENT
