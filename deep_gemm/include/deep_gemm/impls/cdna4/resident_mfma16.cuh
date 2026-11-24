#pragma once
#include <deep_gemm/impls/cdna4/device_common.h>

template<int MF,int NF,int WaveRows,int WaveCols,int Pitch>
__device__ __forceinline__ void finish_resident_mfma16(uint8_t *sm, int wm,int wn,int r,int g,
                                                       const v4f (&acc)[MF][NF]){
    asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)\n\ts_barrier" ::: "memory");
#pragma unroll
    for(int i=0;i<MF;++i)
#pragma unroll
        for(int j=0;j<NF;++j){
            const int offset=(wm*WaveRows+16*i+r)*Pitch+(wn*WaveCols+16*j+4*g)*2;
            const v4f v=acc[i][j];
            *reinterpret_cast<v2u *>(sm+offset)=v2u{pack_bf16x2(v[0],v[1]),pack_bf16x2(v[2],v[3])};
            __builtin_amdgcn_sched_barrier(0);
        }
    asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory");
}

// Operands use the resident kernel's native swizzled LDS stages. This path
// traverses K in 128-element steps and writes the same staged BF16 row tile.
template<int KT,int Stages,int Requests,int Step,int BOff,int SAOff,int SBOff,
         int WaveRows,int WaveCols,int Pitch,class Dma,class Join>
__device__ __forceinline__ void resident_mfma16(uint8_t *sm,int wm,int wn,Dma dma,Join join) {
    constexpr int MF=WaveRows/16,NF=WaveCols/16;
    const int lane=threadIdx.x%64,r=lane%16,g=lane/16;
    v4f acc[MF][NF]{};
    static_for<0,Stages-1>([&](auto I) {
        constexpr int i=decltype(I)::value;
        dma(i,i);
        __builtin_amdgcn_sched_barrier(0);
    });
#pragma unroll 1
    for(int base=0;base<KT;base+=Stages) {
        static_for<0,Stages>([&](auto Slot){
            constexpr int slot=decltype(Slot)::value;
            const int k=base+slot;
            if(k>=KT)return;
            if(k+Stages-2<KT)
                __builtin_amdgcn_s_waitcnt(waitcnt_vm((Stages-2)*Requests));
            else
                __builtin_amdgcn_s_waitcnt(waitcnt_vm(0));
            asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory");
            if(k+Stages-1<KT){
                join(k+Stages-1);
                dma((slot+Stages-1)%Stages,k+Stages-1);
                __builtin_amdgcn_sched_barrier(0);
            }
            const uint8_t *stage=sm+slot*Step;
            v8i a[MF];
            uint32_t code[MF];
#pragma unroll
            for(int i=0;i<MF;++i){
                const int row=wm*WaveRows+16*i+r;
                const int bank=(row>>1)&7;
                const auto *data=stage+row*128;
                const v4i l=*reinterpret_cast<const v4i *>(data+(g^bank)*16);
                const v4i h=*reinterpret_cast<const v4i *>(data+((g+4)^bank)*16);
                a[i]=v8i{l[0],l[1],l[2],l[3],h[0],h[1],h[2],h[3]};
                code[i]=stage[SAOff+row*4+g];
            }
#pragma unroll
            for(int j=0;j<NF;++j){
                const int col=wn*WaveCols+16*j+r;
                const int bank=(col>>2)&3;
                const v4i v=*reinterpret_cast<const v4i *>(stage+BOff+col*64+(g^bank)*16);
                const v8i b{v[0],v[1],v[2],v[3],0,0,0,0};
                const uint32_t bc=stage[SBOff+col*4+g];
#pragma unroll
                for(int i=0;i<MF;++i)
                    acc[i][j]=__builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
                        b,a[i],acc[i][j],4,0,0,bc,0,code[i]);
                __builtin_amdgcn_sched_barrier(0);
            }
        });
    }
    finish_resident_mfma16<MF,NF,WaveRows,WaveCols,Pitch>(sm,wm,wn,r,g,acc);
}

// Retire the current top-left, bottom-left, top-right and bottom-right
// accumulator quadrants in order. Only the first operand quadrant crosses
// the K-loop boundary, leaving registers available for the 128-row tile.
template<int KT,int S,int DMAS,int Step,int BOff,int SAOff,int SBOff,
         int WaveRows,int WaveCols,int Pitch,class DmaOne,class Join>
__device__ __forceinline__ void quadrant_resident_mfma16(uint8_t *sm,int wm,int wn,DmaOne dma_one,Join join){
    constexpr int MF=WaveRows/16,NF=WaveCols/16,HM=MF/2,HN=NF/2,Q=HM*HN;
    static_assert(MF%2==0 && NF%2==0 && Q>=HM+HN && 3*Q>=DMAS);
    const int lane=threadIdx.x%64,r=lane%16,g=lane/16;
    v8i af[MF],bf[NF];
    uint32_t ac[MF],bc[NF];
    v4f acc[MF][NF];
    auto read_a=[&](int i,int slot) {
        const auto *stage=sm+slot*Step;
        const int row=wm*WaveRows+16*i+r,bank=(row>>1)&7;
        const v4i lo=*reinterpret_cast<const v4i *>(stage+row*128+(g^bank)*16);
        const v4i hi=*reinterpret_cast<const v4i *>(stage+row*128+((g+4)^bank)*16);
        af[i]=v8i{lo[0],lo[1],lo[2],lo[3],hi[0],hi[1],hi[2],hi[3]};
        ac[i]=stage[SAOff+row*4+g];
    };
    auto read_b=[&](int j,int slot) {
        const auto *stage=sm+slot*Step;
        const int col=wn*WaveCols+16*j+r,bank=(col>>2)&3;
        const v4i v=*reinterpret_cast<const v4i *>(stage+BOff+col*64+(g^bank)*16);
        bf[j]=v8i{v[0],v[1],v[2],v[3],0,0,0,0};
        bc[j]=stage[SBOff+col*4+g];
    };
    static_for<0,S>([&](auto I){
        constexpr int i=decltype(I)::value;
        static_for<0,DMAS>([&](auto J){dma_one(J,i,i);});
    });
    __builtin_amdgcn_s_waitcnt(waitcnt_vm((S-1)*DMAS));
    asm volatile("s_barrier" ::: "memory");
    static_for<0,HM>([&](auto I){read_a(decltype(I)::value,0);});
    static_for<0,HN>([&](auto J){read_b(decltype(J)::value,0);});
    auto step=[&](int k,auto Slot,auto First,auto Last){
        constexpr int slot=decltype(Slot)::value,next=(slot+1)%S;
        constexpr bool first=decltype(First)::value,last=decltype(Last)::value;
        const int reload=k+S<KT?k+S:k+S-KT;
        if constexpr(!last) join(k+S);
        static_for<0,4*Q>([&](auto T){
            constexpr int t=decltype(T)::value,quad=t/Q;
            constexpr int i=(quad%2)*HM+(t%HM),j=(quad/2)*HN+(t%Q)/HM;
            acc[i][j]=__builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
                bf[j],af[i],first?v4f{}:acc[i][j],4,0,0,bc[j],0,ac[i]);
            if constexpr(t<HM)read_a(HM+t,slot);
            else if constexpr(t<HM+HN)read_b(HN+t-HM,slot);
            if constexpr(!last){
                if constexpr(t==Q-1)asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory");
                if constexpr(t==3*Q){
                    __builtin_amdgcn_s_waitcnt(waitcnt_vm((S-2)*DMAS+(2*DMAS+2)/3));
                    asm volatile("s_barrier" ::: "memory");
                }
                static_for<0,DMAS>([&](auto J){
                    constexpr int d=decltype(J)::value;
                    if constexpr(t==Q+d*3*Q/DMAS)dma_one(J,slot,reload);
                });
                if constexpr(t>=3*Q && t<3*Q+HM)read_a(t-3*Q,next);
                else if constexpr(t>=3*Q+HM && t<3*Q+HM+HN)read_b(t-3*Q-HM,next);
            }
            __builtin_amdgcn_sched_barrier(0);
        });
    };
    step(0,std::integral_constant<int,0>{},std::true_type{},std::false_type{});
#pragma unroll 1
    for(int base=1;base<KT-1;base+=S)
        static_for<0,S>([&](auto J){
            constexpr int j=decltype(J)::value;
            if(base+j<KT-1) step(base+j,std::integral_constant<int,(1+j)%S>{},std::false_type{},std::false_type{});
        });
    step(KT-1,std::integral_constant<int,(KT-1)%S>{},std::false_type{},std::true_type{});
    finish_resident_mfma16<MF,NF,WaveRows,WaveCols,Pitch>(sm,wm,wn,r,g,acc);
}
