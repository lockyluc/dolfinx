// Copyright (C) 2012-2020 Chris N. Richardson, Garth N. Wells and Michal Habera
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "XDMFFile.h"
#include "HDF5File.h"
#include "cells.h"
#include "pugixml.hpp"
#include "xdmf_function.h"
#include "xdmf_mesh.h"
#include "xdmf_meshtags.h"
#include "xdmf_read.h"
#include "xdmf_utils.h"
#include <boost/filesystem.hpp>
#include <boost/lexical_cast.hpp>
#include <dolfinx/common/utils.h>
#include <dolfinx/fem/ElementDofLayout.h>
#include <dolfinx/function/Function.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/graph/Partitioning.h>
#include <dolfinx/mesh/Geometry.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/MeshTags.h>
#include <dolfinx/mesh/Topology.h>
#include <dolfinx/mesh/TopologyComputation.h>

using namespace dolfinx;
using namespace dolfinx::io;

namespace
{
//-----------------------------------------------------------------------------

/// Construct HDF5 filename from XDMF filename
std::string get_hdf5_filename(std::string filename)
{
  boost::filesystem::path p(filename);
  p.replace_extension(".h5");
  if (p.string() == filename)
  {
    throw std::runtime_error("Cannot deduce name of HDF5 file from XDMF "
                             "filename. Filename clash. Check XDMF filename");
  }

  return p.string();
}
//-----------------------------------------------------------------------------

} // namespace

//-----------------------------------------------------------------------------
XDMFFile::XDMFFile(MPI_Comm comm, const std::string filename,
                   const std::string file_mode, const Encoding encoding)
    : _mpi_comm(comm), _filename(filename), _file_mode(file_mode),
      _xml_doc(new pugi::xml_document), _encoding(encoding)
{
  // Check encoding
  if (_encoding == Encoding::ASCII and MPI::size(_mpi_comm.comm()) != 1)
  {
    throw std::runtime_error(
        "Cannot read/write ASCII XDMF in parallel (use HDF5 encoding).");
  }

  // Handle HDF5 and XDMF files with the file mode
  // At the end of this we will have _hdf5_file and _xml_doc
  // both pointing to a valid and opened file handles

  if (_encoding == Encoding::HDF5)
  {
    // See https://www.hdfgroup.org/hdf5-quest.html#gzero on zero for
    // _hdf5_file_id(0)

    const std::string hdf5_filename = get_hdf5_filename(_filename);

    // Open HDF5 file
    const bool mpi_io = MPI::size(_mpi_comm.comm()) > 1 ? true : false;
#ifndef H5_HAVE_PARALLEL
    if (mpi_io)
    {
      throw std::runtime_error(
          "Cannot open file. HDF5 has not been compiled with support for MPI");
    }
#endif
    _h5_id
        = HDF5Interface::open_file(_mpi_comm.comm(), hdf5_filename, file_mode, mpi_io);
    assert(_h5_id > 0);
    LOG(INFO) << "Opened HDF5 file with id \"" << _h5_id << "\"";
  }
  else
    _h5_id = -1;

  if (_file_mode == "r")
  {
    // Load XML doc from file
    pugi::xml_parse_result result = _xml_doc->load_file(_filename.c_str());
    assert(result);

    if (_xml_doc->child("Xdmf").empty())
      throw std::runtime_error("Empty <Xdmf> root node.");

    if (_xml_doc->child("Xdmf").child("Domain").empty())
      throw std::runtime_error("Empty <Domain> node.");
  }
  else if (_file_mode == "w")
  {
    _xml_doc->reset();

    // Add XDMF node and version attribute
    _xml_doc->append_child(pugi::node_doctype)
        .set_value("Xdmf SYSTEM \"Xdmf.dtd\" []");
    pugi::xml_node xdmf_node = _xml_doc->append_child("Xdmf");
    assert(xdmf_node);
    xdmf_node.append_attribute("Version") = "3.0";
    xdmf_node.append_attribute("xmlns:xi") = "http://www.w3.org/2001/XInclude";

    pugi::xml_node domain_node = xdmf_node.append_child("Domain");
    assert(domain_node);
  }
  else if (_file_mode == "a")
  {
    if (boost::filesystem::exists(_filename))
    {
      // Load XML doc from file
      pugi::xml_parse_result result = _xml_doc->load_file(_filename.c_str());
      assert(result);

      if (_xml_doc->child("Xdmf").empty())
        throw std::runtime_error("Empty <Xdmf> root node.");

      if (_xml_doc->child("Xdmf").child("Domain").empty())
        throw std::runtime_error("Empty <Domain> node.");
    }
    else
    {
      _xml_doc->reset();

      // Add XDMF node and version attribute
      _xml_doc->append_child(pugi::node_doctype)
          .set_value("Xdmf SYSTEM \"Xdmf.dtd\" []");
      pugi::xml_node xdmf_node = _xml_doc->append_child("Xdmf");
      assert(xdmf_node);
      xdmf_node.append_attribute("Version") = "3.0";
      xdmf_node.append_attribute("xmlns:xi")
          = "http://www.w3.org/2001/XInclude";

      pugi::xml_node domain_node = xdmf_node.append_child("Domain");
      assert(domain_node);
    }
  }
}
//-----------------------------------------------------------------------------
XDMFFile::~XDMFFile() { close(); }
//-----------------------------------------------------------------------------
void XDMFFile::close()
{
  if (_h5_id > 0)
    HDF5Interface::close_file(_h5_id);
  _h5_id = -1;
}
//-----------------------------------------------------------------------------
void XDMFFile::write_mesh(const mesh::Mesh& mesh, const std::string name,
                          const std::string xpath)
{
  pugi::xml_node node = _xml_doc->select_node(xpath.c_str()).node();

  // Add the mesh Grid to the domain
  xdmf_mesh::add_mesh(_mpi_comm.comm(), node, _h5_id, mesh, name);

  // Save XML file (on process 0 only)
  if (MPI::rank(_mpi_comm.comm()) == 0)
    _xml_doc->save_file(_filename.c_str(), "  ");
}
//-----------------------------------------------------------------------------
void XDMFFile::write_geometry(const mesh::Geometry& geometry,
                              const std::string name, const std::string xpath)
{
  pugi::xml_node node = _xml_doc->select_node(xpath.c_str()).node();

  // Prepare a Grid for Geometry only
  pugi::xml_node grid_node = node.append_child("Grid");
  grid_node.append_attribute("Name") = name.c_str();
  grid_node.append_attribute("GridType") = "Uniform";
  assert(grid_node);

  const std::string path_prefix = "/Geometry/" + name;
  xdmf_mesh::add_geometry_data(_mpi_comm.comm(), grid_node,
                               _h5_id, path_prefix, geometry);

  // Save XML file (on process 0 only)
  if (MPI::rank(_mpi_comm.comm()) == 0)
    _xml_doc->save_file(_filename.c_str(), "  ");
}
//-----------------------------------------------------------------------------
mesh::Mesh XDMFFile::read_mesh(const std::string name,
                               const std::string xpath) const
{
  pugi::xml_node node = _xml_doc->select_node(xpath.c_str()).node();
  pugi::xml_node grid_node
      = node.select_node(("Grid[@Name='" + name + "']").c_str()).node();

  // Read mesh data
  auto [cell_type, x, cells]
      = xdmf_mesh::read_mesh_data(_mpi_comm.comm(), _h5_id, grid_node);

  // TODO: create outside
  // Create a layout
  const fem::ElementDofLayout layout
      = fem::geometry_layout(cell_type, cells.cols());

  // Create Topology
  graph::AdjacencyList<std::int64_t> _cells(cells);
  auto [topology, src, dest] = mesh::create_topology(
      _mpi_comm.comm(), _cells, layout, mesh::GhostMode::none);

  // FIXME: Figure out how to check which entities are required
  // Initialise facet for P2
  // Create local entities
  auto [cell_entity, entity_vertex, index_map]
      = mesh::TopologyComputation::compute_entities(_mpi_comm.comm(), topology,
                                                    1);
  if (cell_entity)
    topology.set_connectivity(cell_entity, topology.dim(), 1);
  if (entity_vertex)
    topology.set_connectivity(entity_vertex, 1, 0);
  if (index_map)
    topology.set_index_map(1, index_map);

  const std::vector<std::int64_t> flags
      = xdmf_mesh::read_flags(_mpi_comm.comm(), _h5_id, grid_node);

  // Create Geometry
  const mesh::Geometry geometry = mesh::create_geometry(
      _mpi_comm.comm(), topology, layout, _cells, dest, src, x, flags);

  // Return Mesh
  return mesh::Mesh(_mpi_comm.comm(), topology, geometry);
}
//-----------------------------------------------------------------------------
std::tuple<
    mesh::CellType,
    Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>,
    Eigen::Array<std::int64_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
XDMFFile::read_mesh_data(const std::string name, const std::string xpath) const
{
  pugi::xml_node node = _xml_doc->select_node(xpath.c_str()).node();
  pugi::xml_node grid_node
      = node.select_node(("Grid[@Name='" + name + "']").c_str()).node();

  return xdmf_mesh::read_mesh_data(_mpi_comm.comm(), _h5_id, grid_node);
}
//-----------------------------------------------------------------------------
void XDMFFile::write_function(const function::Function& function,
                              const double t, const std::string mesh_xpath)
{
  const std::string timegrid_xpath
      = "/Xdmf/Domain/Grid[@GridType='Collection'][@Name='" + function.name
        + "']";
  pugi::xml_node timegrid_node
      = _xml_doc->select_node(timegrid_xpath.c_str()).node();

  if (!timegrid_node)
  {
    pugi::xml_node domain_node = _xml_doc->select_node("/Xdmf/Domain").node();
    timegrid_node = domain_node.append_child("Grid");
    timegrid_node.append_attribute("Name") = function.name.c_str();
    timegrid_node.append_attribute("GridType") = "Collection";
    timegrid_node.append_attribute("CollectionType") = "Temporal";
  }

  pugi::xml_node grid_node = timegrid_node.append_child("Grid");
  grid_node.append_attribute("Name") = function.name.c_str();
  grid_node.append_attribute("GridType") = "Uniform";
  assert(grid_node);

  const std::string ref_path
      = "xpointer(" + mesh_xpath + "/*[self::Topology or self::Geometry])";

  pugi::xml_node topo_geo_ref = grid_node.append_child("xi:include");
  topo_geo_ref.append_attribute("xpointer") = ref_path.c_str();
  assert(topo_geo_ref);

  std::string t_str = boost::lexical_cast<std::string>(t);
  pugi::xml_node time_node = grid_node.append_child("Time");
  time_node.append_attribute("Value") = t_str.c_str();
  assert(time_node);

  // Add the mesh Grid to the domain
  xdmf_function::add_function(_mpi_comm.comm(), function, t, grid_node,
                              _h5_id);

  // Save XML file (on process 0 only)
  if (MPI::rank(_mpi_comm.comm()) == 0)
    _xml_doc->save_file(_filename.c_str(), "  ");
}
//-----------------------------------------------------------------------------
void XDMFFile::write_meshtags(const mesh::MeshTags<int>& meshtags,
                              const std::string geometry_xpath,
                              const std::string xpath)
{
  pugi::xml_node node = _xml_doc->select_node(xpath.c_str()).node();

  pugi::xml_node grid_node = node.append_child("Grid");
  grid_node.append_attribute("Name") = meshtags.name.c_str();
  grid_node.append_attribute("GridType") = "Uniform";
  assert(grid_node);

  const std::string geo_ref_path = "xpointer(" + geometry_xpath + ")";

  pugi::xml_node geo_ref_node = grid_node.append_child("xi:include");
  geo_ref_node.append_attribute("xpointer") = geo_ref_path.c_str();
  assert(geo_ref_node);

  xdmf_meshtags::add_meshtags(_mpi_comm.comm(), meshtags, grid_node, _h5_id,
                              meshtags.name);

  // Save XML file (on process 0 only)
  if (MPI::rank(_mpi_comm.comm()) == 0)
    _xml_doc->save_file(_filename.c_str(), "  ");
}
//-----------------------------------------------------------------------------
mesh::MeshTags<int>
XDMFFile::read_meshtags(std::shared_ptr<const mesh::Mesh> mesh,
                        const std::string name, const std::string xpath,
                        const std::string flags_xpath)
{
  pugi::xml_node node = _xml_doc->select_node(xpath.c_str()).node();
  pugi::xml_node grid_node
      = node.select_node(("Grid[@Name='" + name + "']").c_str()).node();

  pugi::xml_node flags_node = _xml_doc->select_node(flags_xpath.c_str()).node();
  pugi::xml_node topology_node = grid_node.child("Topology");

  // Get topology dataset node
  pugi::xml_node topology_data_node = topology_node.child("DataItem");
  const std::vector<std::int64_t> tdims
      = xdmf_utils::get_dataset_shape(topology_data_node);
  const int nnodes_per_entity = tdims[1];

  // Read topology data
  const std::vector<std::int64_t> topology_data
      = xdmf_read::get_dataset<std::int64_t>(_mpi_comm.comm(), topology_data_node,
                                             _h5_id);
  const std::int32_t num_local_entities = topology_data.size() / nnodes_per_entity;

  const auto cell_type_str = xdmf_utils::get_cell_type(topology_node);
  const mesh::CellType cell_type = mesh::to_type(cell_type_str.first);
  const int e_dim = mesh::cell_dim(cell_type);

  const std::vector<std::int64_t> flags
      = xdmf_mesh::read_flags(_mpi_comm.comm(), _h5_id, flags_node);

  Eigen::Map<const Eigen::Array<std::int64_t, Eigen::Dynamic, 1>> flags_arr(
      flags.data(), flags.size(), 1);

  // Extract only unique topology nodes
  std::vector<std::int64_t> topo_unique;
  topo_unique = topology_data;

  std::sort(topo_unique.begin(), topo_unique.end());
  topo_unique.erase(std::unique(topo_unique.begin(), topo_unique.end()),
                    topo_unique.end());

  // Distribute flags according to unique topology nodes
  const Eigen::Array<std::int64_t, Eigen::Dynamic, 1> dist_flags_arr
      = graph::Partitioning::distribute_data(_mpi_comm.comm(), topo_unique,
                                             flags_arr);

  assert((std::size_t)dist_flags_arr.rows() == topo_unique.size());

  pugi::xml_node values_data_node
      = grid_node.child("Attribute").child("DataItem");

  const std::vector<int> values
      = xdmf_read::get_dataset<int>(_mpi_comm.comm(), values_data_node, _h5_id);

  assert(values.size() == (std::size_t)num_local_entities);

  auto map_e = mesh->topology().index_map(e_dim);
  if (!map_e)
  {
    mesh->create_entities(e_dim);
    map_e = mesh->topology().index_map(e_dim);
  }
  assert(map_e);

  const int dim = mesh->topology().dim();
  const std::int32_t num_entities = map_e->size_local() + map_e->num_ghosts();

  const graph::AdjacencyList<std::int32_t>& cells_g = mesh->geometry().dofmap();

  auto e_to_v = mesh->topology().connectivity(e_dim, 0);
  assert(e_to_v);

  auto e_to_c = mesh->topology().connectivity(e_dim, dim);
  if (!e_to_c)
  {
    mesh->create_connectivity(e_dim, dim);
    e_to_c = mesh->topology().connectivity(e_dim, dim);
    assert(e_to_c);
  }

  auto c_to_v = mesh->topology().connectivity(dim, 0);
  assert(c_to_v);

  std::vector<std::int64_t> entity_flags(nnodes_per_entity);

  const std::vector<std::uint8_t> perm
      = cells::vtk_to_dolfin(cell_type, nnodes_per_entity);

  // Prepare a mapping from *ordered* nodes of entity to entity index
  std::map<std::vector<std::int64_t>, std::int32_t> entities_flags;
  auto map_g = mesh->geometry().index_map();

  // const std::int32_t size_local = map_g->size_local();
  const int size_local_ghosts = map_g->size_local() + map_g->num_ghosts();

  const int rank = MPI::rank(_mpi_comm.comm());

  // const std::vector<std::int64_t>& global_indices = mesh->geometry().global_indices();

  assert(mesh->geometry().flags().size() == (std::size_t)size_local_ghosts);

  const std::vector<std::int64_t> geom_flags(
      mesh->geometry().flags().data(), mesh->geometry().flags().data() + size_local_ghosts);

  const auto ghost_owners = map_g->ghost_owners();
  // Prepare an array of owners of indices to which flags refer by value
  const int comm_size = MPI::size(_mpi_comm.comm());

  // Find the maximum values of flags across all processes
  // Here, must make sure, that a "global" array is accessible with the
  // values of any flag as index
  // const std::int64_t num_flags_global
  //     = MPI::sum(_mpi_comm.comm(), (std::int64_t)size_local_ghosts);

  // Split global array size and retrieve a range that this process is
  // responsible for
  // std::array<std::int64_t, 2> range
  //     = MPI::local_range(_mpi_comm.comm(), num_flags_global);

  std::vector<std::vector<std::int64_t>> send_flags(comm_size);
  std::vector<std::vector<std::int64_t>> recv_flags(comm_size);

  std::vector<std::vector<int>> send_sizes(comm_size);
  std::vector<std::vector<int>> recv_sizes(comm_size);

  for (int i = 0; i < comm_size; ++i)
  {
    send_sizes[i].push_back(size_local_ghosts);
  }

  MPI::all_to_all(_mpi_comm.comm(), send_sizes, recv_sizes);

  for (auto flag : geom_flags)
  {
    // std::int64_t sz = 0;
    // int owner = -1;
    // for (int i = 0; i < comm_size; ++i)
    // {
    //   sz += recv_sizes[i][0];
    //   if (flag < sz)
    //   {
    //     owner = i;
    //     break;
    //   }
    // }
    // const int owner = MPI::index_owner(comm_size, flag, num_flags_global);
    const int owner = MPI::index_owner(comm_size, flag, recv_sizes);
    send_flags[owner].push_back(flag);
  }

  for (std::int32_t e = 0; e < num_entities; ++e)
  {
    // Iterate over all entities of the mesh
    // Find cell attached to the entity
    std::int32_t c = e_to_c->links(e)[0];
    auto cell_nodes = cells_g.links(c);
    auto cell_vertices = c_to_v->links(c);
    auto entity_vertices = e_to_v->links(e);

    for (int v = 0; v < entity_vertices.rows(); ++v)
    {
      // Find local index of vertex wrt. cell
      const int vertex = entity_vertices[perm[v]];
      auto it = std::find(cell_vertices.data(),
                          cell_vertices.data() + cell_vertices.rows(), vertex);
      assert(it != (cell_vertices.data() + cell_vertices.rows()));
      const int local_cell_vertex = std::distance(cell_vertices.data(), it);

      entity_flags[v] = geom_flags[cell_nodes[local_cell_vertex]];

      // const int owner = MPI::index_owner(comm_size, entity_flags[v], num_flags_global);
      // send_flags[owner].push_back(entity_flags[v]);

    }
    std::sort(entity_flags.begin(), entity_flags.end());

    for (int i = 0; i < nnodes_per_entity; ++i)
      std::cout << "proc " << rank << " flag " << entity_flags[i] << std::endl;
    std::cout << std::endl;

    entities_flags.insert({entity_flags, e});
  }

  MPI::all_to_all(_mpi_comm.comm(), send_flags, recv_flags);

  // const int local_size = range[1] - range[0];
  // assert(local_size == size_local_ghosts);

  std::vector<std::int64_t> owners(size_local_ghosts, -1);

  const std::size_t offset
      = MPI::global_offset(_mpi_comm.comm(), size_local_ghosts, true);

  // Iterate over received flags and put owner at the position of the value of flag
  for (int i = 0; i < comm_size; ++i)
  {
    const int num_recv_flags = (int)recv_flags[i].size();
    for (int j = 0; j < num_recv_flags; ++j)
    {
      const int local_index = recv_flags[i][j] - offset;
      assert(size_local_ghosts > local_index);
      assert(local_index >= 0);
      std::cout << "flag " << recv_flags[i][j] << " owner " << i << " loc " << local_index << std::endl;

      owners[local_index] = i;
    }
  }

  const std::vector<std::int64_t> dist_flags(dist_flags_arr.data(),
                                             dist_flags_arr.data() + dist_flags_arr.rows());

  Eigen::Map<Eigen::Array<std::int64_t, Eigen::Dynamic, 1>> owners_arr(
      owners.data(), size_local_ghosts);

  const Eigen::Array<std::int64_t, Eigen::Dynamic, 1> dist_read_flags_owners_arr
      = graph::Partitioning::distribute_data(_mpi_comm.comm(), dist_flags,
                                             owners_arr);

  for (int i = 0; i < (int)dist_flags.size(); ++i)
  {
    std::cout << "proc " << rank << " (" << dist_flags[i] << ", "
              << dist_read_flags_owners_arr(i, 0) << ")" << std::endl;
  }

  // std::cout << "proc" << rank << ": " << flags_owners << std::endl;

  // for (int i = 0; i < flags_owners.rows(); ++i)
  // {
  //   // flags_owners(i, 0) = MPI::index_owner(comm_size, geom_flags[i], num_flags_global);
  //   if (i < size_local)
  //     flags_owners(i, 0) = rank;
  //   else
  //     flags_owners(i, 0) = ghost_owners[i - size_local];
  // }

  // const Eigen::Array<std::int64_t, Eigen::Dynamic, 1> dist_flags_owners_arr
  //     = graph::Partitioning::distribute_data(_mpi_comm.comm(), geom_flags,
  //                                            flags_owners);

  // const Eigen::Array<std::int64_t, Eigen::Dynamic, 1> col_flags_owners_arr
  //     = graph::Partitioning::distribute_data(_mpi_comm.comm(), geom_flags,
  //                                            flags_owners);

  // const Eigen::Array<std::int64_t, Eigen::Dynamic, 1> dist_read_flags_owners_arr
  //     = graph::Partitioning::distribute_data(_mpi_comm.comm(), dist_flags,
  //                                            col_flags_owners_arr);


  Eigen::Map<const Eigen::Array<std::int64_t, Eigen::Dynamic, Eigen::Dynamic,
                                Eigen::RowMajor>>
      entities(topology_data.data(), num_local_entities, nnodes_per_entity);

  std::unordered_map<std::int64_t, std::pair<std::int64_t, std::int64_t>> topo_to_flags;
  for (std::size_t i = 0; i < topo_unique.size(); ++i)
    topo_to_flags[topo_unique[i]] = {dist_flags_arr(i, 0), dist_read_flags_owners_arr(i, 0)};

  std::vector<std::vector<std::int64_t>> send_ents(comm_size);
  std::vector<std::vector<std::int64_t>> recv_ents(comm_size);

  std::vector<std::vector<int>> send_vals(comm_size);
  std::vector<std::vector<int>> recv_vals(comm_size);

  std::vector<std::int64_t> ents(nnodes_per_entity);
  std::vector<int> ents_owners(nnodes_per_entity);

  // Parse entities read from file
  for (Eigen::Index e = 0; e < entities.rows(); ++e)
  {
    std::vector<bool> sent(comm_size, false);

    std::cout << "e " << e << " proc" << rank << " [";
    for (int i = 0; i < nnodes_per_entity; ++i)
    {
      ents[i] = topo_to_flags[entities(e, i)].first;
      ents_owners[i] = (int)topo_to_flags[entities(e, i)].second;
      std::cout << ", (" << ents[i] << ", "<< ents_owners[i] << ")";
    }
    std::cout << "]" << std::endl;

    for (int i = 0; i < nnodes_per_entity; ++i)
    {
      const int send_to = ents_owners[i];
      assert(send_to >= 0);
      if (!sent[send_to])
      {
        send_ents[send_to].insert(send_ents[send_to].end(), ents.begin(), ents.end());
        send_vals[send_to].push_back(values[e]);
        sent[send_to] = true;
      }
    }


  }

  MPI::all_to_all(_mpi_comm.comm(), send_ents, recv_ents);
  MPI::all_to_all(_mpi_comm.comm(), send_vals, recv_vals);

  std::vector<std::int32_t> indices;
  std::vector<int> values_fin;

  // recv_ents[rank].insert(recv_ents[rank].end(), send_ents[rank].begin(), send_ents[rank].end());

  for (int i = 0; i < comm_size; ++i)
  {
    const int num_recv_ents = (int)(recv_ents[i].size() / nnodes_per_entity);
    std::cout << "num recv " << num_recv_ents << " on proc " << rank << "from proc " << i << std::endl;

    for (auto recv : recv_ents[i])
      std::cout << "recv i=" << i << " proc="<<rank<<" val="<<recv<<std::endl;

    for (int e = 0; e < num_recv_ents; ++e)
    {
      std::vector<std::int64_t> flags(&recv_ents[i][nnodes_per_entity * e],
                                      &recv_ents[i][nnodes_per_entity * e]
                                          + nnodes_per_entity);
      int value = recv_vals[i][e];

      std::sort(flags.begin(), flags.end());

      // for (int j = 0; i<nnodes_per_entity;++j)
      // {
      //   std::cout << "proc " << rank << " from " << i <<
      // }

      const auto it = entities_flags.find(flags);
      if (it != entities_flags.end())
      {
        indices.push_back(it->second);
        values_fin.push_back(value);
      }
      else
      {
        for (int j = 0; j < nnodes_per_entity; ++j)
          std::cout << "nf on proc " << rank << ", " << flags[j] << ", ";
        // // std::cout << "proc " << rank << " nf " <<
        // throw std::runtime_error("Entity " + std::to_string(e)
        //                          + " not found in mesh.");
      }
    }
  }

  // for (int e = 0; e < dist_read_entities.num_nodes(); ++e)
  // {
  //   const Eigen::Array<std::int64_t, Eigen::Dynamic, 1> _flags_arr
  //       = dist_read_entities.links(e);

  //   std::vector<std::int64_t> flags(_flags_arr.data(),
  //                                   _flags_arr.data() + _flags_arr.rows());

  //   std::sort(flags.begin(), flags.end());

  //   // std::cout << "Entity on proc " << rank << " " << _flags_arr << std::endl;

  //   const auto it = entities_flags.find(flags);
  //   if (it != entities_flags.end())
  //   {
  //     indices.push_back(it->second);
  //     values_fin.push_back(_values[e]);
  //   }
  //   else
  //   {
  //     std::cout << "e nf proc " << rank << std::endl << _flags_arr << std::endl;
  //     throw std::runtime_error("Entity "+ std::to_string(e) +" not found in mesh.");
  //   }
  // }

  Eigen::Map<const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>> indices_eig(
      indices.data(), indices.size());

  Eigen::Map<const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>> values_eig(
      values_fin.data(), values_fin.size());

  return mesh::MeshTags<int>(mesh, e_dim, indices_eig, values_eig);
}