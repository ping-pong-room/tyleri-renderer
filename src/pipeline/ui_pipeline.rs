use std::io::Cursor;
use std::sync::Arc;

use tyleri_api::data_structure::vertices::{IVertex, UIVertex};
use yarvk::pipeline::color_blend_state::{
    BlendFactor, PipelineColorBlendAttachmentState, PipelineColorBlendStateCreateInfo,
};
use yarvk::pipeline::depth_stencil_state::PipelineDepthStencilStateCreateInfo;
use yarvk::pipeline::input_assembly_state::{
    PipelineInputAssemblyStateCreateInfo, PrimitiveTopology,
};
use yarvk::pipeline::multisample_state::PipelineMultisampleStateCreateInfo;
use yarvk::pipeline::rasterization_state::{PipelineRasterizationStateCreateInfo, PolygonMode};
use yarvk::pipeline::shader_stage::{PipelineShaderStageCreateInfo, ShaderStage};
use yarvk::pipeline::{Pipeline, PipelineCacheType, PipelineLayout, PushConstantRange};
use yarvk::render_pass::RenderPass;
use yarvk::shader_module::ShaderModule;
use yarvk::{
    read_spv, ColorComponentFlags, CompareOp, FrontFace, SampleCountFlags, StencilOp,
    StencilOpState, VertexInputRate,
};

use crate::pipeline::single_image_descriptor_set_layout::SingleImageDescriptorLayout;

pub struct UIPipeline {
    pub pipeline: Arc<Pipeline>,
}

impl UIPipeline {
    pub fn new(
        single_image_descriptor_layout: &SingleImageDescriptorLayout,
        pipeline_cache: PipelineCacheType,
        render_pass: &Arc<RenderPass>,
        subpass: u32,
    ) -> UIPipeline {
        let device = &render_pass.device;
        let mut vertex_spv_file =
            Cursor::new(&include_bytes!(concat!(env!("OUT_DIR"), "/ui.vert"))[..]);
        let mut frag_spv_file =
            Cursor::new(&include_bytes!(concat!(env!("OUT_DIR"), "/ui.frag"))[..]);

        let vertex_code =
            read_spv(&mut vertex_spv_file).expect("Failed to read vertex shader spv file");

        let frag_code =
            read_spv(&mut frag_spv_file).expect("Failed to read fragment shader spv file");

        let vertex_shader_module = ShaderModule::builder(&device, &vertex_code)
            .build()
            .unwrap();

        let fragment_shader_module = ShaderModule::builder(&device, &frag_code).build().unwrap();

        let pipeline_layout = PipelineLayout::builder(&device)
            .add_set_layout(single_image_descriptor_layout.desc_set_layout.clone())
            .add_push_constant_range(
                PushConstantRange::builder()
                    .add_stage(ShaderStage::Vertex)
                    .offset(0)
                    .size(128) // all devices guaranteed to have 128 bytes
                    .build(),
            )
            .build()
            .unwrap();

        let vertex_input_state_info = UIVertex::vertex_input_state(VertexInputRate::VERTEX);
        let noop_stencil_state = StencilOpState {
            fail_op: StencilOp::KEEP,
            pass_op: StencilOp::KEEP,
            depth_fail_op: StencilOp::KEEP,
            compare_op: CompareOp::ALWAYS,
            ..Default::default()
        };

        let entry_name = unsafe { std::ffi::CStr::from_bytes_with_nul_unchecked(b"main\0") };
        // let op_feature = device.get_feature::<{ FeatureType::DeviceFeatures(PhysicalDeviceFeatures::LogicOp) }>().unwrap();
        let pipeline = Pipeline::builder(pipeline_layout)
            .add_stage(
                PipelineShaderStageCreateInfo::builder(vertex_shader_module, entry_name)
                    .stage(ShaderStage::Vertex)
                    .build(),
            )
            .add_stage(
                PipelineShaderStageCreateInfo::builder(fragment_shader_module, entry_name)
                    .stage(ShaderStage::Fragment)
                    .build(),
            )
            .vertex_input_state(vertex_input_state_info)
            .input_assembly_state(
                PipelineInputAssemblyStateCreateInfo::builder()
                    .topology::<{ PrimitiveTopology::TriangleList }>()
                    .build(),
            )
            .rasterization_state(
                PipelineRasterizationStateCreateInfo::builder()
                    .front_face(FrontFace::COUNTER_CLOCKWISE)
                    .line_width(1.0)
                    .polygon_mode(PolygonMode::Fill)
                    .build(),
            )
            .multisample_state(
                PipelineMultisampleStateCreateInfo::builder()
                    .rasterization_samples(SampleCountFlags::TYPE_1)
                    .build(),
            )
            .depth_stencil_state(
                PipelineDepthStencilStateCreateInfo::builder()
                    .depth_test_enable()
                    .depth_write_enable()
                    .depth_compare_op(CompareOp::LESS_OR_EQUAL)
                    .front(noop_stencil_state.clone())
                    .back(noop_stencil_state.clone())
                    .depth_bounds(0.0, 1.0)
                    .build(),
            )
            .color_blend_state(
                PipelineColorBlendStateCreateInfo::builder()
                    .add_attachment(
                        PipelineColorBlendAttachmentState::builder()
                            .src_color_blend_factor(BlendFactor::One)
                            .dst_color_blend_factor(BlendFactor::OneMinusSrcAlpha)
                            // .color_blend_op(BlendOp::ADD)
                            // .src_alpha_blend_factor(BlendFactor::Zero)
                            // .dst_alpha_blend_factor(BlendFactor::Zero)
                            // .alpha_blend_op(BlendOp::ADD)
                            .color_write_mask(ColorComponentFlags::RGBA)
                            .build(),
                    )
                    .build(),
            )
            .cache(pipeline_cache)
            .render_pass(render_pass.clone(), subpass)
            .build()
            .unwrap();
        UIPipeline { pipeline }
    }
}
