// Copyright (C) 2012-2020 Chris N. Richardson, Garth N. Wells and Michal Habera
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "xdmf_function.h"
#include "pugixml.hpp"
#include "xdmf_mesh.h"
#include "xdmf_utils.h"
#include <boost/lexical_cast.hpp>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/fem/DofMap.h>
#include <dolfinx/fem/FiniteElement.h>
#include <dolfinx/function/Function.h>
#include <dolfinx/function/FunctionSpace.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/Topology.h>
#include <string>

using namespace dolfinx;
using namespace dolfinx::io;

namespace
{
// Get data width - normally the same as u.value_size(), but expand for
// 2D vector/tensor because XDMF presents everything as 3D
template <typename T>
int get_padded_width(const function::Function<T>& u)
{
  const int width = u.function_space()->element()->value_size();
  const int rank = u.function_space()->element()->value_rank();
  if (rank == 1 and width == 2)
    return 3;
  else if (rank == 2 and width == 4)
    return 9;
  return width;
}
//-----------------------------------------------------------------------------
template <typename T>
std::vector<T> get_point_data_values(const function::Function<T>& u)
{
  std::shared_ptr<const mesh::Mesh> mesh = u.function_space()->mesh();
  assert(mesh);
  Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> data_values
      = u.compute_point_values();

  const int width = get_padded_width(u);
  assert(mesh->geometry().index_map());
  const int num_local_points = mesh->geometry().index_map()->size_local();
  assert(data_values.rows() >= num_local_points);
  data_values.conservativeResize(num_local_points, Eigen::NoChange);

  // FIXME: Unpick the below code for the new layout of data from
  //        GenericFunction::compute_vertex_values
  std::vector<T> _data_values(width * num_local_points, 0.0);
  const int value_rank = u.function_space()->element()->value_rank();
  if (value_rank > 0)
  {
    // Transpose vector/tensor data arrays
    const int value_size = u.function_space()->element()->value_size();
    for (int i = 0; i < num_local_points; i++)
    {
      for (int j = 0; j < value_size; j++)
      {
        int tensor_2d_offset
            = (j > 1 && value_rank == 2 && value_size == 4) ? 1 : 0;
        _data_values[i * width + j + tensor_2d_offset] = data_values(i, j);
      }
    }
  }
  else
  {
    _data_values = std::vector<T>(
        data_values.data(),
        data_values.data() + data_values.rows() * data_values.cols());
  }

  return _data_values;
}
//-----------------------------------------------------------------------------
template <typename T>
std::vector<T> get_cell_data_values(const function::Function<T>& u)
{
  assert(u.function_space()->dofmap());
  const auto mesh = u.function_space()->mesh();
  const int value_size = u.function_space()->element()->value_size();
  const int value_rank = u.function_space()->element()->value_rank();

  // Allocate memory for function values at cell centres
  const int tdim = mesh->topology().dim();
  const std::int32_t num_local_cells
      = mesh->topology().index_map(tdim)->size_local();
  const std::int32_t local_size = num_local_cells * value_size;

  // Build lists of dofs and create map
  std::vector<std::int32_t> dof_set;
  dof_set.reserve(local_size);
  const auto dofmap = u.function_space()->dofmap();
  assert(dofmap->element_dof_layout);
  const int ndofs = dofmap->element_dof_layout->num_dofs();

  for (int cell = 0; cell < num_local_cells; ++cell)
  {
    // Tabulate dofs
    auto dofs = dofmap->cell_dofs(cell);
    assert(ndofs == value_size);
    for (int i = 0; i < ndofs; ++i)
      dof_set.push_back(dofs[i]);
  }

  // Get values
  std::vector<T> data_values(dof_set.size());
  {
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& x = u.x()->array();
    for (std::size_t i = 0; i < dof_set.size(); ++i)
      data_values[i] = x[dof_set[i]];
  }

  if (value_rank == 1 && value_size == 2)
  {
    // Pad out data for 2D vector to 3D
    data_values.resize(3 * num_local_cells);
    for (int j = (num_local_cells - 1); j >= 0; --j)
    {
      T nd[3] = {data_values[j * 2], data_values[j * 2 + 1], 0};
      std::copy(nd, nd + 3, &data_values[j * 3]);
    }
  }
  else if (value_rank == 2 && value_size == 4)
  {
    data_values.resize(9 * num_local_cells);
    for (int j = (num_local_cells - 1); j >= 0; --j)
    {
      T nd[9] = {data_values[j * 4],
                 data_values[j * 4 + 1],
                 0,
                 data_values[j * 4 + 2],
                 data_values[j * 4 + 3],
                 0,
                 0,
                 0,
                 0};
      std::copy(nd, nd + 9, &data_values[j * 9]);
    }
  }
  return data_values;
}
//-----------------------------------------------------------------------------
// Convert a value_rank to the XDMF string description (Scalar, Vector,
// Tensor)
std::string rank_to_string(int value_rank)
{
  switch (value_rank)
  {
  case 0:
    return "Scalar";
  case 1:
    return "Vector";
  case 2:
    return "Tensor";
  default:
    throw std::runtime_error("Range Error");
  }
}
//-----------------------------------------------------------------------------
/// Returns true for DG0 function::Functions
template <typename T>
bool has_cell_centred_data(const function::Function<T>& u)
{
  int cell_based_dim = 1;
  const int rank = u.function_space()->element()->value_rank();
  for (int i = 0; i < rank; i++)
    cell_based_dim *= u.function_space()->mesh()->topology().dim();

  assert(u.function_space());
  assert(u.function_space()->dofmap());
  assert(u.function_space()->dofmap()->element_dof_layout);
  return (u.function_space()->dofmap()->element_dof_layout->num_dofs()
          == cell_based_dim);
}

template <typename T>
std::vector<std::vector<double>>
get_component_data_values(const std::vector<T>& v)
{
  // FIXME: avoid copy somehow
  std::vector<std::vector<double>> c = {v};
  return c;
}

template <>
std::vector<std::vector<double>>
get_component_data_values(const std::vector<std::complex<double>>& v)
{
  std::vector<std::vector<double>> c(2);
  c[0].resize(v.size());
  c[1].resize(v.size());
  for (std::size_t i = 0; i < v.size(); i++)
    c[0][i] = v[i].real();
  for (std::size_t i = 0; i < v.size(); i++)
    c[1][i] = v[i].imag();
  return c;
}

//-----------------------------------------------------------------------------
} // namespace

//-----------------------------------------------------------------------------
template <typename T>
void xdmf_function::add_function(MPI_Comm comm, const function::Function<T>& u,
                                 const double t, pugi::xml_node& xml_node,
                                 const hid_t h5_id)
{
  LOG(INFO) << "Adding function to node \"" << xml_node.path('/') << "\"";

  assert(u.function_space());
  std::shared_ptr<const mesh::Mesh> mesh = u.function_space()->mesh();
  assert(mesh);

  // Get function::Function data values and shape
  std::vector<T> data_values;
  const bool cell_centred = has_cell_centred_data(u);
  if (cell_centred)
    data_values = get_cell_data_values(u);
  else
    data_values = get_point_data_values(u);

  auto map_c = mesh->topology().index_map(mesh->topology().dim());
  assert(map_c);

  auto map_v = mesh->geometry().index_map();
  assert(map_v);

  // Add attribute DataItem node and write data
  const int width = get_padded_width(u);
  assert(data_values.size() % width == 0);
  const int num_values
      = cell_centred ? map_c->size_global() : map_v->size_global();

  const int value_rank = u.function_space()->element()->value_rank();

  std::vector<std::vector<double>> component_data
      = get_component_data_values(data_values);
  std::vector<std::string> components = {""};
  if (component_data.size() == 2)
    components = {"real", "imag"};

  std::string t_str = boost::lexical_cast<std::string>(t);
  std::replace(t_str.begin(), t_str.end(), '.', '_');

  for (std::size_t i = 0; i < component_data.size(); ++i)
  {
    std::string attr_name;
    if (components[i].empty())
      attr_name = u.name;
    else
      attr_name = components[i] + "_" + u.name;

    std::string dataset_name = "/Function/" + attr_name + "/" + t_str;

    // Add attribute node
    pugi::xml_node attribute_node = xml_node.append_child("Attribute");
    assert(attribute_node);
    attribute_node.append_attribute("Name") = attr_name.c_str();
    attribute_node.append_attribute("AttributeType")
        = rank_to_string(value_rank).c_str();
    attribute_node.append_attribute("Center") = cell_centred ? "Cell" : "Node";

    const bool use_mpi_io = (dolfinx::MPI::size(comm) > 1);

    // Add data item of component
    const std::int64_t offset = dolfinx::MPI::global_offset(
        comm, component_data[i].size() / width, true);
    xdmf_utils::add_data_item(attribute_node, h5_id, dataset_name,
                              component_data[i], offset, {num_values, width},
                              "", use_mpi_io);
  }
}
//-----------------------------------------------------------------------------

// Explicit instantiation
template void xdmf_function::add_function<double>(
    MPI_Comm comm, const function::Function<double>& u, const double t,
    pugi::xml_node& xml_node, const hid_t h5_id);
template void xdmf_function::add_function<std::complex<double>>(
    MPI_Comm comm, const function::Function<std::complex<double>>& u,
    const double t, pugi::xml_node& xml_node, const hid_t h5_id);