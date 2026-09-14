#include "taichi/python/export_vulkan_ray_program.h"
#include "taichi/program/vulkan_ray_program.h"
#include "taichi/program/storage_view.h"
#include "taichi/program/texture.h"
#include "pybind11/stl.h"

namespace taichi {
using namespace lang;
void export_vulkan_ray_program(py::module &m) {
  py::enum_<VulkanRayProgramShader::Stage>(m, "_VulkanRayShaderStage")
      .value("raygen", VulkanRayProgramShader::Stage::raygen)
      .value("miss", VulkanRayProgramShader::Stage::miss)
      .value("closest_hit", VulkanRayProgramShader::Stage::closest_hit)
      .value("any_hit", VulkanRayProgramShader::Stage::any_hit);
  py::enum_<VulkanRayProgramGroup::Kind>(m, "_VulkanRayGroupKind")
      .value("raygen", VulkanRayProgramGroup::Kind::raygen)
      .value("miss", VulkanRayProgramGroup::Kind::miss)
      .value("triangles", VulkanRayProgramGroup::Kind::triangles);
  py::class_<VulkanRayProgramShader>(m, "_VulkanRayProgramShader")
      .def(py::init<>())
      .def_readwrite("stage", &VulkanRayProgramShader::stage)
      .def_readwrite("words", &VulkanRayProgramShader::words)
      .def_readwrite("entry", &VulkanRayProgramShader::entry);
  py::class_<VulkanRayProgramGroup>(m, "_VulkanRayProgramGroup")
      .def(py::init<>())
      .def_readwrite("kind", &VulkanRayProgramGroup::kind)
      .def_readwrite("general", &VulkanRayProgramGroup::general)
      .def_readwrite("closest_hit", &VulkanRayProgramGroup::closest_hit)
      .def_readwrite("any_hit", &VulkanRayProgramGroup::any_hit);
  py::class_<VulkanRayProgramRecord>(m, "_VulkanRayProgramRecord")
      .def(py::init<>())
      .def_readwrite("group", &VulkanRayProgramRecord::group)
      .def_readwrite("data", &VulkanRayProgramRecord::data);
  py::class_<VulkanRayProgramBuffer>(m, "_VulkanRayProgramBuffer")
      .def(py::init<>())
      .def_readwrite("set", &VulkanRayProgramBuffer::set)
      .def_readwrite("binding", &VulkanRayProgramBuffer::binding)
      .def_readwrite("storage", &VulkanRayProgramBuffer::storage)
      .def_readwrite("uniform", &VulkanRayProgramBuffer::uniform)
      .def_readwrite("writable", &VulkanRayProgramBuffer::writable);
  py::class_<VulkanRayProgramImage>(m, "_VulkanRayProgramImage")
      .def(py::init<>())
      .def_readwrite("set", &VulkanRayProgramImage::set)
      .def_readwrite("binding", &VulkanRayProgramImage::binding)
      .def_readwrite("texture", &VulkanRayProgramImage::texture)
      .def_readwrite("storage", &VulkanRayProgramImage::storage)
      .def_readwrite("mip_level", &VulkanRayProgramImage::mip_level);
  py::class_<VulkanRayProgramAS>(m, "_VulkanRayProgramAS")
      .def(py::init<>())
      .def_readwrite("set", &VulkanRayProgramAS::set)
      .def_readwrite("binding", &VulkanRayProgramAS::binding)
      .def_readwrite("handle", &VulkanRayProgramAS::handle);
  py::class_<VulkanRayProgramLaunch>(m, "_VulkanRayProgramLaunch")
      .def(py::init<>())
      .def_readwrite("dimensions", &VulkanRayProgramLaunch::dimensions)
      .def_readwrite("raygen", &VulkanRayProgramLaunch::raygen)
      .def_readwrite("miss", &VulkanRayProgramLaunch::miss)
      .def_readwrite("hit", &VulkanRayProgramLaunch::hit)
      .def_readwrite("buffers", &VulkanRayProgramLaunch::buffers)
      .def_readwrite("images", &VulkanRayProgramLaunch::images)
      .def_readwrite("scenes", &VulkanRayProgramLaunch::scenes)
      .def_readwrite("push_constants", &VulkanRayProgramLaunch::push_constants);
  py::class_<PreparedVulkanRayLaunch, std::shared_ptr<PreparedVulkanRayLaunch>>(
      m, "_PreparedVulkanRayLaunch");
}
}  // namespace taichi
