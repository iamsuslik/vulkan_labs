#include "application.hpp"
#include <imgui.h>
#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>
#include <algorithm>

namespace application {

    struct Vertex {
        float position[3];
        float color[3];
    };

    struct ObjectUBO {
        float mvp[16];
        float tintColor[4];
    };

    struct Mat4 {
        float m[16] = { 0 };

        static Mat4 identity() {
            Mat4 r;
            r.m[0] = 1.0f; r.m[5] = 1.0f; r.m[10] = 1.0f; r.m[15] = 1.0f;
            return r;
        }

        static Mat4 multiply(const Mat4& a, const Mat4& b) {
            Mat4 r;
            for (int c = 0; c < 4; ++c) {
                for (int row = 0; row < 4; ++row) {
                    float sum = 0.0f;
                    for (int k = 0; k < 4; ++k) sum += a.m[k * 4 + row] * b.m[c * 4 + k];
                    r.m[c * 4 + row] = sum;
                }
            }
            return r;
        }

        static Mat4 translate(float tx, float ty, float tz) {
            Mat4 r = identity();
            r.m[12] = tx; r.m[13] = ty; r.m[14] = tz;
            return r;
        }

        static Mat4 scale(float sx, float sy, float sz) {
            Mat4 r;
            r.m[0] = sx; r.m[5] = sy; r.m[10] = sz; r.m[15] = 1.0f;
            return r;
        }

        static Mat4 rotateX(float rad) {
            Mat4 r = identity();
            float c = cosf(rad), s = sinf(rad);
            r.m[5] = c;  r.m[6] = s;
            r.m[9] = -s; r.m[10] = c;
            return r;
        }

        static Mat4 rotateY(float rad) {
            Mat4 r = identity();
            float c = cosf(rad), s = sinf(rad);
            r.m[0] = c;  r.m[2] = -s;
            r.m[8] = s;  r.m[10] = c;
            return r;
        }

        static Mat4 rotateZ(float rad) {
            Mat4 r = identity();
            float c = cosf(rad), s = sinf(rad);
            r.m[0] = c;  r.m[1] = s;
            r.m[4] = -s; r.m[5] = c;
            return r;
        }

        // Перспектива 
        static Mat4 perspective(float fovRad, float aspect, float zNear, float zFar) {
            Mat4 r;
            float f = 1.0f / tanf(fovRad * 0.5f);
            r.m[0] = f / aspect;
            r.m[5] = f; 
            r.m[10] = zFar / (zFar - zNear);
            r.m[11] = 1.0f;
            r.m[14] = -(zNear * zFar) / (zFar - zNear);
            return r;
        }

        // Ортография
        static Mat4 ortho(float left, float right, float bottom, float top, float zNear, float zFar) {
            Mat4 r;
            r.m[0] = 2.0f / (right - left);
            r.m[5] = 2.0f / (top - bottom);
            r.m[10] = 1.0f / (zFar - zNear);
            r.m[12] = -(right + left) / (right - left);
            r.m[13] = -(bottom + top) / (bottom - top);
            r.m[14] = -zNear / (zFar - zNear);
            r.m[15] = 1.0f;
            return r;
        }
    };

    namespace {
        VkBuffer vk_vertex_buffer = VK_NULL_HANDLE;
        VmaAllocation vk_vertex_buffer_allocation = VK_NULL_HANDLE;

        VkBuffer vk_index_buffer = VK_NULL_HANDLE;
        VmaAllocation vk_index_buffer_allocation = VK_NULL_HANDLE;
        uint32_t indices_count = 0;

        VkPipelineLayout vk_pipeline_layout = VK_NULL_HANDLE;
        VkPipeline vk_pipeline = VK_NULL_HANDLE;

        VkDescriptorSetLayout vk_descriptor_set_layout = VK_NULL_HANDLE;
        VkDescriptorPool vk_descriptor_pool = VK_NULL_HANDLE;

        // Объект 1 (Основной)
        VkBuffer vk_ubo_buffer1 = VK_NULL_HANDLE;
        VmaAllocation vk_ubo_alloc1 = VK_NULL_HANDLE;
        VkDescriptorSet vk_descriptor_set1 = VK_NULL_HANDLE;

        // Объект 2 
        VkBuffer vk_ubo_buffer2 = VK_NULL_HANDLE;
        VmaAllocation vk_ubo_alloc2 = VK_NULL_HANDLE;
        VkDescriptorSet vk_descriptor_set2 = VK_NULL_HANDLE;

        // Переменные интерфейса
        int projection_type = 0; // 0 - Perspective, 1 - Orthographic
        float position[3] = { -0.8f, 0.0f, 0.0f }; 
        float rotation[3] = { 0.4f, 0.0f, 0.0f };  
        float scale_val[3] = { 1.0f, 1.0f, 1.0f };

        bool animate = false;
        float anim_speed = 1.0f;
        float anim_radius = 0.8f;
        double accumulated_time = 0.0;

        float tint_color[3] = { 1.0f, 1.0f, 1.0f };
        bool show_second_object = true;

        VkShaderModule loadShaderModule(const char* path, VkDevice device) {
            std::ifstream file(path, std::ios::binary | std::ios::ate);
            if (!file.is_open()) return VK_NULL_HANDLE;
            size_t size = file.tellg();
            std::vector<uint32_t> buffer(size / sizeof(uint32_t));
            file.seekg(0);
            file.read(reinterpret_cast<char*>(buffer.data()), size);
            file.close();

            VkShaderModuleCreateInfo createInfo = {
                .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                .codeSize = size,
                .pCode = buffer.data(),
            };

            VkShaderModule shaderModule;
            if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) return VK_NULL_HANDLE;
            return shaderModule;
        }
    }

    bool initialize() {
        auto& context = graphics::internal::context;

        // ГЕОМЕТРИЯ КОНУСА И ПРОЦЕДУРНЫЕ ЦВЕТА
        std::vector<Vertex> vertices;
        std::vector<uint32_t> indices;
        int segments = 36;
        float height = 1.4f;
        float radius = 0.6f;

        auto calc_color = [&](float x, float y, float z) -> std::vector<float> {
            float r = (x + radius) / (2.0f * radius);
            float g = (y + height * 0.5f) / height;
            float b = (z + radius) / (2.0f * radius);
            return { std::clamp(r, 0.1f, 1.0f), std::clamp(g, 0.1f, 1.0f), std::clamp(b, 0.1f, 1.0f) };
            };

        // Верхушка конуса
        auto top_c = calc_color(0.0f, height * 0.5f, 0.0f);
        vertices.push_back({ { 0.0f, height * 0.5f, 0.0f }, { top_c[0], top_c[1], top_c[2] } });

        // Центр дна
        auto bot_c = calc_color(0.0f, -height * 0.5f, 0.0f);
        vertices.push_back({ { 0.0f, -height * 0.5f, 0.0f }, { bot_c[0], bot_c[1], bot_c[2] } });

        // Окружность основания
        for (int i = 0; i < segments; ++i) {
            float angle = 2.0f * 3.14159265f * static_cast<float>(i) / static_cast<float>(segments);
            float x = radius * cos(angle);
            float z = radius * sin(angle);
            float y = -height * 0.5f;
            auto c = calc_color(x, y, z);
            vertices.push_back({ { x, y, z }, { c[0], c[1], c[2] } });
        }

        for (int i = 0; i < segments; ++i) {
            uint32_t curr = 2 + i;
            uint32_t next = 2 + ((i + 1) % segments);
            indices.push_back(0); indices.push_back(curr); indices.push_back(next);
        }
        for (int i = 0; i < segments; ++i) {
            uint32_t curr = 2 + i;
            uint32_t next = 2 + ((i + 1) % segments);
            indices.push_back(1); indices.push_back(next); indices.push_back(curr);
        }
        indices_count = static_cast<uint32_t>(indices.size());

        // БУФЕРЫ VRAM 
        VkBufferCreateInfo vb_info = {
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size = sizeof(Vertex) * vertices.size(),
            .usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        };
        VmaAllocationCreateInfo alloc_info = {
            .flags = VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
            .usage = VMA_MEMORY_USAGE_AUTO,
        };
        if (vmaCreateBuffer(context.allocator, &vb_info, &alloc_info, &vk_vertex_buffer, &vk_vertex_buffer_allocation, nullptr) != VK_SUCCESS)
            return false;

        Vertex* vb_data = nullptr;
        vmaMapMemory(context.allocator, vk_vertex_buffer_allocation, (void**)&vb_data);
        memcpy(vb_data, vertices.data(), sizeof(Vertex) * vertices.size());
        vmaUnmapMemory(context.allocator, vk_vertex_buffer_allocation);

        VkBufferCreateInfo ib_info = {
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size = sizeof(uint32_t) * indices.size(),
            .usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        };
        if (vmaCreateBuffer(context.allocator, &ib_info, &alloc_info, &vk_index_buffer, &vk_index_buffer_allocation, nullptr) != VK_SUCCESS) {
            shutdown(); return false;
        }
        uint32_t* ib_data = nullptr;
        vmaMapMemory(context.allocator, vk_index_buffer_allocation, (void**)&ib_data);
        memcpy(ib_data, indices.data(), sizeof(uint32_t) * indices.size());
        vmaUnmapMemory(context.allocator, vk_index_buffer_allocation);

        // ДЕСКРИПТОРЫ И UBO
        VkDescriptorSetLayoutBinding uboLayoutBinding = {
            .binding = 0,
            .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
        };
        VkDescriptorSetLayoutCreateInfo layoutInfo = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .bindingCount = 1,
            .pBindings = &uboLayoutBinding,
        };
        if (vkCreateDescriptorSetLayout(context.device, &layoutInfo, nullptr, &vk_descriptor_set_layout) != VK_SUCCESS) {
            shutdown(); return false;
        }

        VkDescriptorPoolSize poolSize = { .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .descriptorCount = 2 };
        VkDescriptorPoolCreateInfo poolInfo = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .maxSets = 2,
            .poolSizeCount = 1,
            .pPoolSizes = &poolSize,
        };
        if (vkCreateDescriptorPool(context.device, &poolInfo, nullptr, &vk_descriptor_pool) != VK_SUCCESS) {
            shutdown(); return false;
        }

        VkBufferCreateInfo ubo_info = {
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size = sizeof(ObjectUBO),
            .usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        };
        vmaCreateBuffer(context.allocator, &ubo_info, &alloc_info, &vk_ubo_buffer1, &vk_ubo_alloc1, nullptr);
        vmaCreateBuffer(context.allocator, &ubo_info, &alloc_info, &vk_ubo_buffer2, &vk_ubo_alloc2, nullptr);

        VkDescriptorSetLayout layouts[] = { vk_descriptor_set_layout, vk_descriptor_set_layout };
        VkDescriptorSetAllocateInfo allocSetInfo = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool = vk_descriptor_pool,
            .descriptorSetCount = 2,
            .pSetLayouts = layouts,
        };
        VkDescriptorSet sets[2];
        vkAllocateDescriptorSets(context.device, &allocSetInfo, sets);
        vk_descriptor_set1 = sets[0];
        vk_descriptor_set2 = sets[1];

        VkDescriptorBufferInfo bInfo1 = { .buffer = vk_ubo_buffer1, .offset = 0, .range = sizeof(ObjectUBO) };
        VkDescriptorBufferInfo bInfo2 = { .buffer = vk_ubo_buffer2, .offset = 0, .range = sizeof(ObjectUBO) };
        VkWriteDescriptorSet descriptorWrites[] = {
            {
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = vk_descriptor_set1,
                .dstBinding = 0,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                .pBufferInfo = &bInfo1,
            },
            {
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = vk_descriptor_set2,
                .dstBinding = 0,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                .pBufferInfo = &bInfo2,
            }
        };
        vkUpdateDescriptorSets(context.device, 2, descriptorWrites, 0, nullptr);

        // Пайплан
        VkShaderModule vertShader = loadShaderModule("shaders/shader.vert.spv", context.device);
        VkShaderModule fragShader = loadShaderModule("shaders/shader.frag.spv", context.device);
        if (!vertShader || !fragShader) {
            if (vertShader) vkDestroyShaderModule(context.device, vertShader, nullptr);
            if (fragShader) vkDestroyShaderModule(context.device, fragShader, nullptr);
            shutdown(); return false;
        }

        VkPipelineShaderStageCreateInfo shaderStages[] = {
            {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_VERTEX_BIT, .module = vertShader, .pName = "main" },
            {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_FRAGMENT_BIT, .module = fragShader, .pName = "main" }
        };

        VkVertexInputBindingDescription bindingDesc = { .binding = 0, .stride = sizeof(Vertex), .inputRate = VK_VERTEX_INPUT_RATE_VERTEX };
        VkVertexInputAttributeDescription attribDesc[2] = {
            {.location = 0, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT, .offset = offsetof(Vertex, position) },
            {.location = 1, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT, .offset = offsetof(Vertex, color) }
        };
        VkPipelineVertexInputStateCreateInfo vertexInputInfo = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
            .vertexBindingDescriptionCount = 1, .pVertexBindingDescriptions = &bindingDesc,
            .vertexAttributeDescriptionCount = 2, .pVertexAttributeDescriptions = attribDesc
        };

        VkPipelineInputAssemblyStateCreateInfo inputAssembly = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
            .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST
        };
        VkPipelineViewportStateCreateInfo viewportState = { .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, .viewportCount = 1, .scissorCount = 1 };
        VkPipelineRasterizationStateCreateInfo rasterizer = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
            .polygonMode = VK_POLYGON_MODE_FILL,
            .cullMode = VK_CULL_MODE_NONE,
            .lineWidth = 1.0f
        };
        VkPipelineMultisampleStateCreateInfo multisampling = { .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO, .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT };
        VkPipelineDepthStencilStateCreateInfo depthStencil = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
            .depthTestEnable = VK_TRUE, .depthWriteEnable = VK_TRUE, .depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL
        };
        VkPipelineColorBlendAttachmentState colorBlendAttachment = {
            .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT
        };
        VkPipelineColorBlendStateCreateInfo colorBlending = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO, .attachmentCount = 1, .pAttachments = &colorBlendAttachment
        };
        std::vector<VkDynamicState> dynamicStates = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
        VkPipelineDynamicStateCreateInfo dynamicState = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
            .dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()), .pDynamicStates = dynamicStates.data()
        };

        VkPipelineLayoutCreateInfo pipelineLayoutInfo = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .setLayoutCount = 1,
            .pSetLayouts = &vk_descriptor_set_layout,
        };
        if (vkCreatePipelineLayout(context.device, &pipelineLayoutInfo, nullptr, &vk_pipeline_layout) != VK_SUCCESS) {
            vkDestroyShaderModule(context.device, vertShader, nullptr);
            vkDestroyShaderModule(context.device, fragShader, nullptr);
            shutdown(); return false;
        }

        VkGraphicsPipelineCreateInfo pipelineInfo = {
            .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
            .stageCount = 2, .pStages = shaderStages,
            .pVertexInputState = &vertexInputInfo, .pInputAssemblyState = &inputAssembly,
            .pViewportState = &viewportState, .pRasterizationState = &rasterizer,
            .pMultisampleState = &multisampling, .pDepthStencilState = &depthStencil,
            .pColorBlendState = &colorBlending, .pDynamicState = &dynamicState,
            .layout = vk_pipeline_layout, .renderPass = context.render_pass,
        };
        if (vkCreateGraphicsPipelines(context.device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &vk_pipeline) != VK_SUCCESS) {
            vkDestroyShaderModule(context.device, vertShader, nullptr);
            vkDestroyShaderModule(context.device, fragShader, nullptr);
            shutdown(); return false;
        }

        vkDestroyShaderModule(context.device, vertShader, nullptr);
        vkDestroyShaderModule(context.device, fragShader, nullptr);
        return true;
    }

    void shutdown() {
        auto& context = graphics::internal::context;
        vkQueueWaitIdle(context.graphics_queue);

        if (vk_pipeline) { vkDestroyPipeline(context.device, vk_pipeline, nullptr); vk_pipeline = VK_NULL_HANDLE; }
        if (vk_pipeline_layout) { vkDestroyPipelineLayout(context.device, vk_pipeline_layout, nullptr); vk_pipeline_layout = VK_NULL_HANDLE; }
        if (vk_descriptor_pool) { vkDestroyDescriptorPool(context.device, vk_descriptor_pool, nullptr); vk_descriptor_pool = VK_NULL_HANDLE; }
        if (vk_descriptor_set_layout) { vkDestroyDescriptorSetLayout(context.device, vk_descriptor_set_layout, nullptr); vk_descriptor_set_layout = VK_NULL_HANDLE; }

        if (context.allocator != VK_NULL_HANDLE) {
            if (vk_ubo_buffer1) { vmaDestroyBuffer(context.allocator, vk_ubo_buffer1, vk_ubo_alloc1); vk_ubo_buffer1 = VK_NULL_HANDLE; }
            if (vk_ubo_buffer2) { vmaDestroyBuffer(context.allocator, vk_ubo_buffer2, vk_ubo_alloc2); vk_ubo_buffer2 = VK_NULL_HANDLE; }
            if (vk_vertex_buffer) { vmaDestroyBuffer(context.allocator, vk_vertex_buffer, vk_vertex_buffer_allocation); vk_vertex_buffer = VK_NULL_HANDLE; }
            if (vk_index_buffer) { vmaDestroyBuffer(context.allocator, vk_index_buffer, vk_index_buffer_allocation); vk_index_buffer = VK_NULL_HANDLE; }
        }
    }

    void update(double dt) {
        if (animate) accumulated_time += dt * anim_speed;

        ImGui::Begin("3D Control Panel", nullptr, ImGuiWindowFlags_AlwaysAutoResize);

        // Task 1: Projection
        ImGui::Text("1. Projection:");
        ImGui::RadioButton("Perspective", &projection_type, 0); ImGui::SameLine();
        ImGui::RadioButton("Orthographic", &projection_type, 1);
        ImGui::Separator();

        // Task 2: Transforms
        ImGui::Text("2. Object 1 Transforms:");
        ImGui::DragFloat3("Position", position, 0.02f, -5.0f, 5.0f);
        ImGui::DragFloat3("Rotation", rotation, 0.05f, -6.28f, 6.28f);
        ImGui::DragFloat3("Scale", scale_val, 0.02f, 0.1f, 3.0f);
        ImGui::Separator();

        // Task 3: Animation
        ImGui::Text("3. Trajectory Animation:");
        ImGui::Checkbox("Play Animation", &animate);
        ImGui::SliderFloat("Speed", &anim_speed, 0.1f, 5.0f);
        ImGui::SliderFloat("Radius", &anim_radius, 0.2f, 2.0f);
        ImGui::Separator();

        // Task 4: Tint Color
        ImGui::Text("4. Color Tint (Multiplies Vertices):");
        ImGui::ColorEdit3("Tint Color", tint_color);
        ImGui::Separator();

        // Task 6: Multi-objects
        ImGui::Text("6. Multi-Objects (VkDescriptorSet):");
        ImGui::Checkbox("Show Object 2", &show_second_object);

        ImGui::End();

        // матрицы
        float aspect = (float)graphics::internal::context.swapchain_extent.width / (float)graphics::internal::context.swapchain_extent.height;
        Mat4 proj = (projection_type == 0)
            ? Mat4::perspective(1.2f, aspect, 0.1f, 100.0f)
            : Mat4::ortho(-1.5f * aspect, 1.5f * aspect, -1.5f, 1.5f, -10.0f, 10.0f);

        // Камеру отодвигаем на 4.5 метра назад
        Mat4 view = Mat4::translate(0.0f, 0.0f, 4.5f);

        // ОБЪЕКТ 1 (Управляемый)
        float cur_x = position[0];
        float cur_y = position[1];
        float cur_z = position[2];
        float cur_rot_y = rotation[1];

        if (animate) {
            float t = static_cast<float>(accumulated_time);
            cur_x += anim_radius * cosf(t);
            cur_y += (anim_radius * 0.5f) * sinf(2.0f * t);
            cur_rot_y += t * 2.0f;
        }

        Mat4 model1 = Mat4::multiply(Mat4::translate(cur_x, cur_y, cur_z),
            Mat4::multiply(Mat4::rotateZ(rotation[2]),
                Mat4::multiply(Mat4::rotateY(cur_rot_y),
                    Mat4::multiply(Mat4::rotateX(rotation[0]),
                        Mat4::scale(scale_val[0], scale_val[1], scale_val[2])))));

        Mat4 mvp1 = Mat4::multiply(proj, Mat4::multiply(view, model1));

        ObjectUBO ubo1{};
        memcpy(ubo1.mvp, mvp1.m, sizeof(float) * 16);
        ubo1.tintColor[0] = tint_color[0]; ubo1.tintColor[1] = tint_color[1]; ubo1.tintColor[2] = tint_color[2]; ubo1.tintColor[3] = 1.0f;

        vmaCopyMemoryToAllocation(graphics::internal::context.allocator, &ubo1, vk_ubo_alloc1, 0, sizeof(ObjectUBO));

        // ОБЪЕКТ 2 (Справа на экране, вращается сам, зеленый оттенок)
        if (show_second_object) {
            float t2 = static_cast<float>(accumulated_time * 1.5);
            Mat4 model2 = Mat4::multiply(Mat4::translate(1.0f, 0.0f, 0.0f), // Сдвинут вправо
                Mat4::multiply(Mat4::rotateX(0.4f), // Наклонен
                    Mat4::multiply(Mat4::rotateY(t2), // Крутится
                        Mat4::scale(0.7f, 0.7f, 0.7f)))); // Чуть меньше размером

            Mat4 mvp2 = Mat4::multiply(proj, Mat4::multiply(view, model2));

            ObjectUBO ubo2{};
            memcpy(ubo2.mvp, mvp2.m, sizeof(float) * 16);
            ubo2.tintColor[0] = 0.2f; ubo2.tintColor[1] = 1.0f; ubo2.tintColor[2] = 0.4f; ubo2.tintColor[3] = 1.0f;

            vmaCopyMemoryToAllocation(graphics::internal::context.allocator, &ubo2, vk_ubo_alloc2, 0, sizeof(ObjectUBO));
        }
    }

    void render(const graphics::internal::FrameData& fd) {
        VkCommandBufferBeginInfo beginInfo = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT };
        vkBeginCommandBuffer(fd.command_buffer, &beginInfo);

        VkClearValue clearValues[2] = {};
        clearValues[0].color = { { 0.08f, 0.08f, 0.1f, 1.0f } };
        clearValues[1].depthStencil = { 1.0f, 0 };

        VkRenderPassBeginInfo renderPassInfo = {
            .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
            .renderPass = graphics::internal::context.render_pass,
            .framebuffer = fd.framebuffer,
            .renderArea = {.offset = {0, 0}, .extent = graphics::internal::context.swapchain_extent },
            .clearValueCount = 2,
            .pClearValues = clearValues
        };
        vkCmdBeginRenderPass(fd.command_buffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

        VkViewport viewport = { .x = 0, .y = 0, .width = (float)graphics::internal::context.swapchain_extent.width, .height = (float)graphics::internal::context.swapchain_extent.height, .minDepth = 0.0f, .maxDepth = 1.0f };
        vkCmdSetViewport(fd.command_buffer, 0, 1, &viewport);

        VkRect2D scissor = { .offset = {0, 0}, .extent = graphics::internal::context.swapchain_extent };
        vkCmdSetScissor(fd.command_buffer, 0, 1, &scissor);

        vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, vk_pipeline);

        VkDeviceSize offset = 0;
        vkCmdBindVertexBuffers(fd.command_buffer, 0, 1, &vk_vertex_buffer, &offset);
        vkCmdBindIndexBuffer(fd.command_buffer, vk_index_buffer, 0, VK_INDEX_TYPE_UINT32);

        // Рисуем Объект 1 (слева)
        vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, vk_pipeline_layout, 0, 1, &vk_descriptor_set1, 0, nullptr);
        vkCmdDrawIndexed(fd.command_buffer, indices_count, 1, 0, 0, 0);

        // Рисуем Объект 2 (справа)
        if (show_second_object) {
            vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, vk_pipeline_layout, 0, 1, &vk_descriptor_set2, 0, nullptr);
            vkCmdDrawIndexed(fd.command_buffer, indices_count, 1, 0, 0, 0);
        }

        vkCmdEndRenderPass(fd.command_buffer);
        vkEndCommandBuffer(fd.command_buffer);
    }

} // namespace application