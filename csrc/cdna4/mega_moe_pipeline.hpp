#pragma once

#include <torch/extension.h>
#include <ATen/hip/impl/HIPStreamMasqueradingAsCUDA.h>
#include <c10/hip/HIPGuard.h>
#include <algorithm>
#include <cstdlib>
#include <optional>
#include <sstream>
#include <vector>
#include <array>
#include <map>
#include "jit_runtime.hpp"
#include "deep_gemm/impls/cdna4/mega_moe_pipeline_args.h"

namespace deep_gemm::cdna4::nv_pipeline_host {

inline thread_local int dense_cu_limit=0;
inline thread_local torch::Tensor task_trace,communication_trace,worker_trace;

inline int option(const char* key,int fallback) {
    const char* value=std::getenv(key);return value?std::atoi(value):fallback;
}

inline thread_local std::map<int,std::array<hipEvent_t,7>> diagnostic_events;
inline void phase(int index,hipStream_t stream) {
    if(!option("DG_MEGA_MOE_STAGE1_EVENTS",0))return;
    int device=0;TORCH_CHECK(hipGetDevice(&device)==hipSuccess);
    auto& events=diagnostic_events[device];
    if(!events[0])for(auto& e:events)TORCH_CHECK(hipEventCreate(&e)==hipSuccess);
    // Ordinary recorded events are capture dependencies only: their timing remains
    // from the last eager call. External event nodes refresh timestamps on replay.
    TORCH_CHECK(hipEventRecordWithFlags(events[index],stream,hipEventRecordExternal)==hipSuccess);
}
inline std::vector<double> phase_times() {
    int device=0;TORCH_CHECK(hipGetDevice(&device)==hipSuccess);
    auto it=diagnostic_events.find(device);TORCH_CHECK(it!=diagnostic_events.end());
    TORCH_CHECK(hipEventSynchronize(it->second[6])==hipSuccess);
    std::vector<double> times;
    for(int i=1;i<7;++i){float ms=0;TORCH_CHECK(hipEventElapsedTime(&ms,it->second[i-1],it->second[i])==hipSuccess);times.push_back(ms);}
    return times;
}

inline GemmArgs matrix(const uint8_t* input,const int32_t* input_sf,const torch::Tensor& weight,
                      const torch::Tensor& weight_sf,void* output,int rows,int columns,int inner,
                      int sf_pitch,bool packed) {
    TORCH_CHECK(weight.stride(-1)==1 && weight.stride(-2)==inner/(packed?2:1),
                "NV-style pipeline requires contiguous K-major weight rows");
    TORCH_CHECK(weight_sf.scalar_type()==torch::kInt32 && weight_sf.stride(-2)==1);
    GemmArgs a{};
    a.A=input;a.B=reinterpret_cast<const uint8_t*>(weight.data_ptr());a.sfa=input_sf;
    a.sfb=weight_sf.data_ptr<int32_t>();a.D=output;a.M=rows;a.N=columns;a.K=inner;
    a.lda=inner;a.ldb=inner/(packed?2:1);a.lda_sf=sf_pitch;a.ldb_sf=weight_sf.stride(-1);
    a.ldd=columns;a.alpha=1;a.splits=1;a.m_alignment=256;
    a.b_group_stride=weight.dim()==3?weight.stride(0):0;
    a.sfb_group_stride=weight_sf.dim()==3?weight_sf.stride(0):0;
    return a;
}

inline void constants(std::ostringstream& s,const char* name,const GemmArgs& a) {
    s<<"#define "<<name<<"(a) do { ";
    if(a.K) {
        s<<"(a).N="<<a.N<<"; (a).K="<<a.K<<"; (a).lda_sf="<<a.lda_sf<<"; (a).ldb_sf="<<a.ldb_sf<<"; "
         <<"(a).b_group_stride="<<a.b_group_stride<<"LL; (a).sfb_group_stride="<<a.sfb_group_stride<<"LL; "
         <<"(a).moe_record_stride="<<a.moe_record_stride<<"; (a).moe_weight_word="<<a.moe_weight_word<<"; "
         <<"(a).moe_sf_pitch="<<a.moe_sf_pitch<<"; (a).moe_fast_math="<<a.moe_fast_math<<"; ";
    }
    s<<"} while (0)\n";
}

// Dense is the existing dense-GEMM entry point. Default kernels and their ABI
// remain unchanged; this opt-in path owns only its queue/workspace arguments.
template<class Dense>
inline void run(MegaMoeArgs p,const torch::Tensor& buffer,const torch::Tensor& y,
                const torch::Tensor& w1,const torch::Tensor& sf1,const torch::Tensor& w2,const torch::Tensor& sf2,
                const std::optional<torch::Tensor>& sw1,const std::optional<torch::Tensor>& ssf1,
                const std::optional<torch::Tensor>& sw2,const std::optional<torch::Tensor>& ssf2,
                const torch::Tensor& l1_out,const torch::Tensor& l2_out,
                const torch::Tensor& shared_l1_out,const torch::Tensor& shared_out,Dense&& dense) {
    const auto& l=p.layout;const int tokens=p.num_tokens,cus=num_compute_units();
    const bool shared=sw1.has_value() && tokens>0;
    const bool stage1_only=option("DG_MEGA_MOE_STAGE1_QUEUE",0)!=0;
    TORCH_CHECK(!stage1_only || !sw1.has_value(),"Stage1-only queue currently requires no shared expert");
    TORCH_CHECK(l.with_sf && w1.size(-1)*2==l.hidden && w2.size(-1)*2==l.intermediate,
                "NV-style pipeline supports FP8 activations and packed FP4 routed weights");
    TORCH_CHECK(l.hidden%256==0 && l.intermediate%256==0 && l.hidden>=256 && l.intermediate>=256);
    TORCH_CHECK(!shared || l.shared_intermediate==l.intermediate,"pipeline supports one shared expert");
    TORCH_CHECK(cus>=32 && l.intermediate/128<=64,"pipeline requires at least 32 CUs and intermediate <= 8192");
    const int comm=std::clamp(option("DG_MOE_PIPELINE_COMM_CUS",32),8,cus-16);
    const int l1_cus=std::clamp(option("DG_MOE_PIPELINE_L1_CUS",144)/8*8,8,cus-comm-8);
    const int cache=option("DG_MOE_PIPELINE_CACHE",0);
    const bool stream_k=!stage1_only && option("DG_MOE_PIPELINE_STREAM_K",0)!=0;
    const bool fixed=option("DG_MOE_PIPELINE_FIXED_STAGES",1)!=0;
    const bool split_shared=shared && option("DG_MOE_PIPELINE_SHARED_SPLIT",1)!=0;
    const bool profile=option("DG_MOE_PIPELINE_PROFILE",0)!=0;
    TORCH_CHECK(cus>=32 && (cache==0 || cache==2));
    TORCH_CHECK(!stream_k || l.intermediate/128<=64);
    TORCH_CHECK(!(stream_k && cache==2),"cached payload epochs currently require whole-block L2 readiness");
    p.ordered_push=1;p.direct_combine=0;
    const auto stream=c10::hip::getCurrentHIPStream(y.get_device()).stream();
    if(stage1_only)phase(0,stream);
    auto* base=static_cast<uint8_t*>(buffer.data_ptr());
    const auto ptr=[&](int64_t off){return base+off;};
    // The legacy full-layer queue borrows output-row storage for arrival flags.
    // Stage1-only uses a proven-unused activation tail so it can prepare output
    // pointers concurrently, preserving the existing symmetric-buffer ABI.
    const bool early_pointers=option("DG_MEGA_MOE_STAGE1_EARLY_POINTERS",stage1_only?1:0)!=0;
    int64_t flag_offset=l.output_rows;
    size_t flag_bytes=((l.max_rows+127)/128+l.num_experts_per_rank+l.num_ranks)*sizeof(int);
    TORCH_CHECK(flag_bytes<=size_t(l.max_rows)*sizeof(int64_t));
    if(early_pointers) {
        TORCH_CHECK(stage1_only && l.num_topk<=l.num_experts_per_rank);
        const int64_t routed=int64_t(l.num_max_tokens)*l.num_ranks*std::min(l.num_topk,l.num_experts_per_rank);
        // Sum of aligned expert lengths is an alignment multiple and <= this
        // floor bound. The existing layout reserves the ceil bound, leaving
        // a whole unused block for the target shape; no valid payload aliases it.
        const int64_t upper=(routed+int64_t(l.num_experts_per_rank)*(l.alignment-1))/l.alignment*l.alignment;
        flag_bytes=((l.max_rows+255)/256)*sizeof(int);
        TORCH_CHECK(upper<l.max_rows && int64_t(flag_bytes)<=(l.max_rows-upper)*l.hidden*l.act_bytes,"No provably unused activation tail for arrival flags");
        flag_offset=l.l1_acts+upper*l.hidden*l.act_bytes;
    }
    TORCH_CHECK(hipMemsetAsync(ptr(flag_offset),0,flag_bytes,stream)==hipSuccess);

    const bool compact_barrier=option("DG_MEGA_MOE_COMPACT_BARRIER",stage1_only?1:0)!=0;
    auto service=jit_runtime.get_module("nv_pipeline_services:"+std::to_string(compact_barrier),[&]{return std::string(
        "#define DG_MOE_COMPACT_BARRIER ")+std::to_string(compact_barrier)+"\n"
        "#include <deep_gemm/impls/cdna4/mega_moe.cuh>\n"
        "#define K(name,body) extern \"C\" __global__ __launch_bounds__(256) void nv_##name(MegaMoeArgs p){body;}\n"
        "K(route,mega_moe_route_impl(p))\nK(publish,mega_moe_barrier_impl<true>(p))\n"
        "K(finish,mega_moe_finish_impl(p))\nK(swiglu,mega_moe_swiglu_impl<true>(p))\n"
        "K(reduce,mega_moe_reduce_impl(p))\n";},
        {"nv_route","nv_publish","nv_finish","nv_swiglu","nv_reduce"});
    const auto launch=[&](int k,MegaMoeArgs a,int blocks,int threads,int lds,hipStream_t on){
        TORCH_CHECK(JitRuntime::launch_kernel(service->kernels[k],a,dim3(blocks),dim3(threads),lds,on)==0);
    };
    static thread_local std::vector<std::pair<hipEvent_t,hipEvent_t>> events;
    hipEvent_t shared_ready=nullptr;
    if(split_shared) {
        const int device=y.get_device();if(events.size()<=size_t(device))events.resize(device+1,{nullptr,nullptr});
        auto& pair=events[device];
        if(!pair.first) {
            TORCH_CHECK(hipEventCreateWithFlags(&pair.first,hipEventDisableTiming)==hipSuccess);
            TORCH_CHECK(hipEventCreateWithFlags(&pair.second,hipEventDisableTiming)==hipSuccess);
        }
        auto side=at::hip::getStreamFromPoolMasqueradingAsCUDA(false,device);
        if(side.stream()==stream)side=at::hip::getStreamFromPoolMasqueradingAsCUDA(false,device);
        TORCH_CHECK(side.stream()!=stream);
        TORCH_CHECK(hipEventRecord(pair.first,stream)==hipSuccess);
        TORCH_CHECK(hipStreamWaitEvent(side.stream(),pair.first,0)==hipSuccess);
        shared_l1_out.record_stream(side);shared_out.record_stream(side);
        const at::hip::HIPStreamGuardMasqueradingAsCUDA guard(side);
        auto sh=p;sh.shared_part=1;
        launch(2,sh,std::min((tokens+3)/4,4*cus),256,0,side.stream());
        auto acts=torch::from_blob(ptr(l.x),{tokens,l.hidden},buffer.options().dtype(torch::kFloat8_e4m3fn));
        auto scales=torch::from_blob(ptr(l.shared_l1_sf),{tokens,l.hidden/128},{1,l.shared_sf_pitch},buffer.options().dtype(torch::kInt32));
        {struct Restore{int value;~Restore(){dense_cu_limit=value;}} restore{dense_cu_limit};
         dense_cu_limit=std::max(1,cus-8);dense(acts,scales,*sw1,*ssf1,shared_l1_out);}
        launch(3,sh,std::min((tokens+3)/4,4*cus),256,0,side.stream());
        TORCH_CHECK(hipEventRecord(pair.second,side.stream())==hipSuccess);shared_ready=pair.second;
    }

    MegaMoePipelineArgs q{};q.moe=p;q.shared_tokens=shared?tokens:0;q.dispatch_blocks=comm;
    q.l1_workers=l1_cus;q.n1=l.intermediate/128;q.n2=l.hidden/256;
    q.max_blocks=(l.max_rows+255)/256+(tokens+255)/256;
    q.min_l1_ready=std::clamp(q.n1-option("DG_MOE_PIPELINE_L2_LOOKAHEAD",2),1,q.n1);
    auto blocks=torch::empty({q.max_blocks,int(sizeof(MegaMoeRowBlock)/sizeof(int))},y.options().dtype(torch::kInt32));
    auto sends=torch::empty({(tokens*l.num_topk+63)/64+2*l.num_experts,4},y.options().dtype(torch::kInt32));
    auto state=torch::empty({544+4*q.max_blocks+2*q.max_blocks*q.n2+8*q.max_blocks},y.options().dtype(torch::kInt32));
    auto ready=torch::empty({q.max_blocks},y.options().dtype(torch::kInt64));
    auto context=torch::empty({int64_t(sizeof(q))},buffer.options());
    q.blocks=reinterpret_cast<MegaMoeRowBlock*>(blocks.data_ptr<int>());
    q.push_blocks=reinterpret_cast<MegaMoePushBlock*>(sends.data_ptr<int>());
    q.context=reinterpret_cast<MegaMoePipelineArgs*>(context.data_ptr());
    q.state=state.data_ptr<int>();q.l1_done=q.state+544;q.l2_done=q.l1_done+q.max_blocks;
    q.sf_ready=q.l2_done+q.max_blocks;q.l1_issued=q.sf_ready+q.max_blocks;
    q.jobs=q.l1_issued+q.max_blocks;q.job_ready=q.jobs+q.max_blocks*q.n2;
    q.combine_jobs=q.job_ready+q.max_blocks*q.n2;q.combine_ready=q.combine_jobs+q.max_blocks*4;
    q.k_ready=reinterpret_cast<uint64_t*>(ready.data_ptr<int64_t>());
    q.l1=matrix(ptr(l.l1_acts),reinterpret_cast<int32_t*>(ptr(l.l1_sf)),w1,sf1,l1_out.data_ptr(),l.max_rows,2*l.intermediate,l.hidden,l.sf_pitch,true);
    q.l2=matrix(ptr(l.l2_acts),reinterpret_cast<int32_t*>(ptr(l.l2_sf)),w2,sf2,l2_out.data_ptr(),l.max_rows,l.hidden,l.intermediate,l.sf_pitch,true);
    q.l1.moe_acts=ptr(l.l2_acts);q.l1.moe_sf=reinterpret_cast<int32_t*>(ptr(l.l2_sf));
    q.l1.moe_records=reinterpret_cast<int32_t*>(ptr(l.l1_records));q.l1.moe_record_stride=l.record_ints;
    q.l1.moe_weight_word=l.hidden/128;q.l1.moe_sf_pitch=l.sf_pitch;q.l1.moe_clamp=p.activation_clamp;q.l1.moe_fast_math=p.fast_math;
    if(shared) {
        q.shared_l1=matrix(ptr(l.x),reinterpret_cast<int32_t*>(ptr(l.shared_l1_sf)),*sw1,*ssf1,shared_l1_out.data_ptr(),tokens,2*l.intermediate,l.hidden,l.shared_sf_pitch,false);
        q.shared_l2=matrix(ptr(l.shared_l2_acts),reinterpret_cast<int32_t*>(ptr(l.shared_l2_sf)),*sw2,*ssf2,shared_out.data_ptr(),tokens,l.hidden,l.intermediate,l.shared_sf_pitch,false);
        q.shared_l1.moe_acts=ptr(l.shared_l2_acts);q.shared_l1.moe_sf=reinterpret_cast<int32_t*>(ptr(l.shared_l2_sf));
        q.shared_l1.moe_sf_pitch=l.shared_sf_pitch;q.shared_l1.moe_clamp=p.activation_clamp;q.shared_l1.moe_fast_math=p.fast_math;
    }
    if(profile) {
        task_trace=torch::zeros({int64_t(q.max_blocks)*(q.n1+q.n2),4,12},y.options().dtype(torch::kInt64));
        communication_trace=torch::zeros({int64_t(q.max_blocks)*5+sends.size(0),4,8},y.options().dtype(torch::kInt64));
        worker_trace=torch::zeros({4,cus,4},y.options().dtype(torch::kInt64));
        q.metrics=reinterpret_cast<uint64_t*>(task_trace.data_ptr<int64_t>());
        q.comm_metrics=reinterpret_cast<uint64_t*>(communication_trace.data_ptr<int64_t>());
        q.worker_ids=reinterpret_cast<uint64_t*>(worker_trace.data_ptr<int64_t>());
    }
    const bool local_only=option("DG_MEGA_MOE_STAGE1_LOCAL_ONLY",stage1_only?1:0)!=0;
    TORCH_CHECK(!local_only || (stage1_only && cache==0));
    const bool fast_init=option("DG_MEGA_MOE_STAGE1_FAST_INIT",stage1_only?1:0)!=0;
    const bool lite_publish=option("DG_MEGA_MOE_STAGE1_LITE_PUBLISH",stage1_only?1:0)!=0;
    TORCH_CHECK(!(fast_init||lite_publish) || (stage1_only && l.num_experts_per_rank<=256));
    const int send_rows=option("DG_MOE_PIPELINE_SEND_ROWS",stage1_only?256:64);
    TORCH_CHECK(send_rows==64||send_rows==128||send_rows==256);
    std::ostringstream code;
    code<<"#define DG_MOE_EARLY_POINTERS "<<early_pointers<<"\n#define DG_MOE_INPUT_FLAG_OFFSET "<<flag_offset<<"LL\n";
    code<<"#define DG_MOE_L1_LOCAL_ONLY "<<local_only<<"\n";
    code<<"#define DG_MOE_FAST_INIT "<<fast_init<<"\n#define DG_MOE_L1_ONLY_PUBLISH "<<lite_publish<<"\n";
    code<<"#define DG_MOE_SEND_ROWS "<<send_rows<<"\n";
    code<<"#define DG_MOE_STAGE1_ONLY "<<stage1_only<<"\n";
    code<<"#define DG_MOE_TASK_M 256\n#define DG_MOE_PUSH_BLOCKS 1\n#define DG_MOE_BLOCK_PIPELINE 1\n"
          "#define DG_MOE_COMM_SF 1\n#define DG_MOE_RELOAD_ARGS 1\n#define DG_MOE_SHARED_TAIL 1\n#define DG_MOE_EARLY_READY "+std::to_string(!stage1_only)+"\n"
        <<"#define DG_MOE_SHARED_SPLIT "<<split_shared<<"\n#define DG_MOE_STREAM_K "<<stream_k
        <<"\n#define DG_MOE_CACHE_A "<<cache<<"\n#define DG_MOE_QUEUE_PROFILE "<<profile
        <<"\n#define DG_MOE_COPY_BATCH "<<option("DG_MOE_PIPELINE_COPY_BATCH",0)
        <<"\n#define DG_MOE_NATIVE_ALIAS_L2 "<<option("DG_MOE_PIPELINE_ALIAS_L2",1)
        <<"\n#define DG_MOE_L1_N "<<2*l.intermediate<<"\n#define DG_MOE_L1_K "<<l.hidden
        <<"\n#define DG_MOE_L2_N "<<l.hidden<<"\n#define DG_MOE_L2_K "<<l.intermediate<<"\n";
    constants(code,"NV_CONST_L1",q.l1);constants(code,"NV_CONST_L2",q.l2);
    constants(code,"NV_CONST_S1",q.shared_l1);constants(code,"NV_CONST_S2",q.shared_l2);
    code<<"#include <deep_gemm/impls/cdna4/mega_moe_pipeline.cuh>\n"
          "using Q=deep_gemm::cdna4::MegaMoePipelineArgs;\n"
          "extern \"C\" __global__ __launch_bounds__(256) void nv_init(Q q){nv_pipeline::init(q);}\n"
          "extern \"C\" __global__ __attribute__((amdgpu_flat_work_group_size(256,256),amdgpu_waves_per_eu(1,1))) void nv_resident(Q q){"
          ;
    if(stage1_only)code<<"if(nv_pipeline::is_dispatch(q))nv_pipeline::dispatch(q);nv_pipeline::compute<0>(q);";
    else {code<<"if(blockIdx.x<q.dispatch_blocks){nv_pipeline::dispatch(q);return;}";
    if(fixed)code<<"if(blockIdx.x<q.dispatch_blocks+q.l1_workers)nv_pipeline::compute<0>(q);else nv_pipeline::compute<1>(q);";
    else code<<"nv_pipeline::compute(q);";
    }
    code<<"}\nextern \"C\" __global__ __launch_bounds__(256) void nv_wait(Q q){nv_pipeline::wait_outputs(q);}\n";
    const auto source=code.str();auto module=jit_runtime.get_module("nv_pipeline_"+std::to_string(std::hash<std::string>{}(source)),[&]{return source;},{"nv_init","nv_resident","nv_wait"});
    launch(0,p,std::max(1,std::min((tokens*l.num_topk+255)/256,4*cus)),256,0,stream);
    launch(1,p,1,256,l.num_ranks*l.num_experts*4,stream);
    TORCH_CHECK(JitRuntime::launch_kernel(module->kernels[0],q,dim3(1),dim3(256),0,stream)==0);
    if(shared_ready){TORCH_CHECK(hipStreamWaitEvent(stream,shared_ready,0)==hipSuccess);}
    constexpr unsigned lds=152064;int occupancy=0;
    TORCH_CHECK(hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy,module->kernels[1],256,lds)==hipSuccess && occupancy>=1,
                "NV-style pipeline needs one resident workgroup per CU");
    if(stage1_only)phase(1,stream);
    TORCH_CHECK(JitModule::launch_function(module->kernels[1],dim3(cus),dim3(256),lds,stream,true,q)==hipSuccess);
    if(stage1_only){phase(2,stream);return;}
    TORCH_CHECK(JitRuntime::launch_kernel(module->kernels[2],q,dim3(1),dim3(128),0,stream)==0);
    if(tokens)launch(4,p,std::min((tokens+3)/4,4*cus),256,0,stream);
}
} // namespace deep_gemm::cdna4::nv_pipeline_host
