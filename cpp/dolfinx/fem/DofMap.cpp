// Copyright (C) 2007-2018 Anders Logg and Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "DofMap.h"
#include "DofMapBuilder.h"
#include "ElementDofLayout.h"
#include "utils.h"
#include <cstdint>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/MPI.h>
#include <dolfinx/common/types.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/mesh/Topology.h>

using namespace dolfinx;
using namespace dolfinx::fem;

namespace
{
template <typename T>
Eigen::Array<std::int32_t, Eigen::Dynamic, 1>
remap_dofs(const std::vector<std::int32_t>& old_to_new,
           const graph::AdjacencyList<T>& dofs_old)
{
  const Eigen::Array<T, Eigen::Dynamic, 1>& _dofs_old = dofs_old.array();
  Eigen::Array<std::int32_t, Eigen::Dynamic, 1> dofmap(_dofs_old.rows());
  for (Eigen::Index i = 0; i < dofmap.size(); ++i)
    dofmap[i] = old_to_new[_dofs_old[i]];
  return dofmap;
}
//-----------------------------------------------------------------------------
// Build a new, collapsed DofMap directly from a dofmap view by simple
// re-indexing
fem::DofMap build_collapsed_dofmap(MPI_Comm comm, const DofMap& dofmap_view,
                                   const mesh::Topology& topology)
{
  auto element_dof_layout = std::make_shared<ElementDofLayout>(
      dofmap_view.element_dof_layout->copy());
  assert(element_dof_layout);

  // FIXME X
  if (dofmap_view.bs() == 1 and element_dof_layout->block_size() > 1)
  {
    throw std::runtime_error(
        "Cannot collapse dofmap with block size greater than 1 from parent "
        "with block size of 1. Create new dofmap first.");
  }

  // FIXME X
  if (dofmap_view.bs() > 1 and element_dof_layout->block_size() > 1)
  {
    throw std::runtime_error(
        "Cannot (yet) collapse dofmap with block size greater than 1 from "
        "parent with block size greater than 1. Create new dofmap first.");
  }

  if (dofmap_view.bs() != 1)
    throw("Untested");

  // Get topological dimension
  const int tdim = topology.dim();
  auto cells = topology.connectivity(tdim, 0);
  assert(cells);

  // Build set of dof indicies that are in the new dofmap
  std::vector<std::int32_t> dofs_in_view;
  for (int i = 0; i < cells->num_nodes(); ++i)
  {
    auto cell_dofs = dofmap_view.cell_dofs(i);
    for (Eigen::Index dof = 0; dof < cell_dofs.rows(); ++dof)
      dofs_in_view.push_back(cell_dofs[dof]);
  }
  std::sort(dofs_in_view.begin(), dofs_in_view.end());
  dofs_in_view.erase(std::unique(dofs_in_view.begin(), dofs_in_view.end()),
                     dofs_in_view.end());

  // FIXME X
  // Get block sizes
  // const int bs_view = dofmap_view.element_dof_layout->block_size();
  // const int bs_map_view = dofmap_view.bs();

  // TODO: throw is view block size != 1

  // Compute sizes (owned and un-owned by this rank)
  const std::int32_t num_owned_view
      = dofmap_view.index_map->size_local() * dofmap_view.index_map_bs();
  const auto it_unowned0 = std::lower_bound(dofs_in_view.begin(),
                                            dofs_in_view.end(), num_owned_view);
  const std::size_t num_owned
      = std::distance(dofs_in_view.begin(), it_unowned0);
  const std::size_t num_unowned
      = std::distance(it_unowned0, dofs_in_view.end());

  // Get rank offset for new dofmap
  const std::int64_t process_offset
      = dolfinx::MPI::global_offset(comm, num_owned, true);

  // std::cout << "Map Block size: " << dofmap_view.index_map_bs() << std::endl;
  // std::cout << "Map size: " << dofmap_view.index_map->size_local() <<
  // std::endl; std::cout << "Parent bs: " << dofmap_view.bs() << std::endl;

  const int map_bs = dofmap_view.index_map_bs();

  // For owned dofs, compute new global index
  std::vector<std::int64_t> global_index(dofmap_view.index_map->size_local(),
                                         -1);
  for (auto it = dofs_in_view.begin(); it != it_unowned0; ++it)
  {
    const std::size_t index_new = std::distance(dofs_in_view.begin(), it);
    assert(*it / map_bs < (int)global_index.size());
    global_index[*it / map_bs] = index_new + process_offset;
  }

  // Send new global indices for owned dofs to non-owning process, and
  // receive new global indices from owner
  const std::vector global_index_remote
      = dofmap_view.index_map->scatter_fwd(global_index, 1);
  const Eigen::Array<std::int32_t, Eigen::Dynamic, 1> ghost_owner_old
      = dofmap_view.index_map->ghost_owner_rank();

  // Compute ghosts for collapsed dofmap
  Eigen::Array<std::int64_t, Eigen::Dynamic, 1> ghosts(num_unowned);
  std::vector<int> ghost_owners(num_unowned);
  for (auto it = it_unowned0; it != dofs_in_view.end(); ++it)
  {
    const std::int32_t index = std::distance(it_unowned0, it);
    const std::int32_t index_old = (*it - num_owned_view) / map_bs;
    assert(global_index_remote[index_old] >= 0);

    assert(index < (std::int32_t)ghosts.size());
    assert(index_old < (std::int32_t)global_index_remote.size());
    ghosts[index] = global_index_remote[index_old];

    assert(index < (std::int32_t)ghost_owners.size());
    assert(index_old < (std::int32_t)ghost_owner_old.size());
    ghost_owners[index] = ghost_owner_old[index_old];
  }

  std::cout << "Collapse 6" << std::endl;

  // Create new index map
  // FIXME X
  auto index_map = std::make_shared<common::IndexMap>(
      comm, num_owned,
      dolfinx::MPI::compute_graph_edges(
          comm, std::set<int>(ghost_owners.begin(), ghost_owners.end())),
      ghosts, ghost_owners);

  // Create array from dofs in view to new dof indices
  std::vector<std::int32_t> old_to_new(dofs_in_view.back() + 1, -1);
  std::int32_t count = 0;
  for (auto& dof : dofs_in_view)
    old_to_new[dof] = count++;

  // Build new dofmap
  const graph::AdjacencyList<std::int32_t>& dof_array_view = dofmap_view.list();
  Eigen::Array<std::int32_t, Eigen::Dynamic, 1> dofmap
      = remap_dofs(old_to_new, dof_array_view);

  // Dimension sanity checks
  assert(element_dof_layout);
  assert(dofmap.rows()
         == (cells->num_nodes() * element_dof_layout->num_dofs()));

  const int cell_dimension = element_dof_layout->num_dofs();
  assert(dofmap.rows() % cell_dimension == 0);
  Eigen::Map<Eigen::Array<std::int32_t, Eigen::Dynamic, Eigen::Dynamic,
                          Eigen::RowMajor>>
      _dofmap(dofmap.data(), dofmap.rows() / cell_dimension, cell_dimension);

  return fem::DofMap(element_dof_layout, index_map,
                     element_dof_layout->block_size(),
                     graph::AdjacencyList<std::int32_t>(_dofmap),
                     element_dof_layout->block_size());
}

} // namespace

//-----------------------------------------------------------------------------
graph::AdjacencyList<std::int32_t>
fem::transpose_dofmap(graph::AdjacencyList<std::int32_t>& dofmap,
                      std::int32_t num_cells)
{
  // Count number of cell contributions to each global index
  const std::int32_t max_index
      = dofmap.array().head(dofmap.offsets()(num_cells)).maxCoeff();
  std::vector<int> num_local_contributions(max_index + 1, 0);
  for (int c = 0; c < num_cells; ++c)
  {
    auto dofs = dofmap.links(c);
    for (int i = 0; i < dofs.rows(); ++i)
      num_local_contributions[dofs[i]]++;
  }

  // Compute offset for each global index
  std::vector<int> index_offsets(num_local_contributions.size() + 1, 0);
  std::partial_sum(num_local_contributions.begin(),
                   num_local_contributions.end(), index_offsets.begin() + 1);

  std::vector<std::int32_t> data(index_offsets.back());
  std::vector<int> pos = index_offsets;
  int cell_offset = 0;
  for (int c = 0; c < num_cells; ++c)
  {
    auto dofs = dofmap.links(c);
    for (int i = 0; i < dofs.rows(); ++i)
      data[pos[dofs[i]]++] = cell_offset++;
  }

  // Sort the source indices for each global index
  // This could improve linear memory access
  // FIXME: needs profiling
  for (int index = 0; index < max_index; ++index)
  {
    std::sort(data.begin() + index_offsets[index],
              data.begin() + index_offsets[index + 1]);
  }

  return graph::AdjacencyList<std::int32_t>(data, index_offsets);
}
//-----------------------------------------------------------------------------
DofMap::DofMap(std::shared_ptr<const ElementDofLayout> element_dof_layout,
               std::shared_ptr<const common::IndexMap> index_map,
               int index_map_bs,
               const graph::AdjacencyList<std::int32_t>& dofmap, int bs)
    : element_dof_layout(element_dof_layout), index_map(index_map),
      _index_map_bs(index_map_bs), _dofmap(dofmap), _bs(bs)
{
  // Dofmap data is copied as the types for dofmap and _dofmap may
  // differ, typically 32- vs 64-bit integers
}
//-----------------------------------------------------------------------------
DofMap DofMap::extract_sub_dofmap(const std::vector<int>& component) const
{
  assert(!component.empty());

  // Get sub-element dof layout
  assert(element_dof_layout);
  std::shared_ptr<const ElementDofLayout> sub_element_dof_layout
      = this->element_dof_layout->sub_dofmap(component);

  const int bs_sub = sub_element_dof_layout->block_size();

  // Get components in parent map that correspond to sub-dofs
  const std::vector sub_element_map_view
      = this->element_dof_layout->sub_view(component);

  // FIXME X: Handle blocked sub-dofs from blocked parent maps
  if (this->element_dof_layout->block_size() > 1 and bs_sub != 1)
    throw std::runtime_error("extract_sub_dofmap needs updating");

  // Build dofmap by extracting from parent
  const int bs = this->element_dof_layout->block_size();
  const int num_cells = this->_dofmap.num_nodes();

  // std::cout << "!!!!!extract: " << bs << std::endl;

  // FIXME X: is this dofs per cell or (dofs per cell)/bs
  const std::int32_t sub_dofs_per_cell = sub_element_map_view.size();
  Eigen::Array<std::int32_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      dofmap(num_cells, sub_dofs_per_cell);
  for (int c = 0; c < num_cells; ++c)
  {
    auto cell_dmap_parent = this->_dofmap.links(c);
    for (std::int32_t i = 0; i < sub_dofs_per_cell; ++i)
    {
      const std::div_t pos = std::div(sub_element_map_view[i], _index_map_bs);
      // std::cout << "q, r: " << pos.quot << ", " << pos.rem << std::endl;
      // std::cout << "dof:  " << bs * cell_dmap_parent[pos.quot] + pos.rem
      //           << std::endl;
      dofmap(c, i) = bs * cell_dmap_parent[pos.quot] + pos.rem;
    }
  }

  return DofMap(sub_element_dof_layout, this->index_map, this->_index_map_bs,
                graph::AdjacencyList<std::int32_t>(dofmap), 1);
}
//-----------------------------------------------------------------------------
std::pair<std::unique_ptr<DofMap>, std::vector<std::int32_t>>
DofMap::collapse(MPI_Comm comm, const mesh::Topology& topology) const
{
  assert(element_dof_layout);
  assert(index_map);

  // Create new element dof layout and reset parent
  std::unique_ptr<DofMap> dofmap_new;

  if (this->_index_map_bs == 1 and this->element_dof_layout->block_size() > 1)
  {
    // Parent does not have block structure but sub-map does, so build
    // new submap to get block structure for collapsed dofmap.
    // Create new dofmap

    // Copy dof layout
    auto collapsed_dof_layout
        = std::make_shared<ElementDofLayout>(element_dof_layout->copy());

    // Build new dofmap data
    auto [index_map, dofmap]
        = DofMapBuilder::build(comm, topology, *collapsed_dof_layout);

    // Create new dofmap with block size
    dofmap_new = std::make_unique<DofMap>(
        element_dof_layout, index_map, collapsed_dof_layout->block_size(),
        std::move(dofmap), collapsed_dof_layout->block_size());
  }
  else
  {
    // FIXME X
    // Collapse dof map, without building and re-ordering from scratch
    // std::cout << "Here I am" << std::endl;
    dofmap_new = std::make_unique<DofMap>(
        build_collapsed_dofmap(comm, *this, topology));
    std::cout << "Post Here I am: " << dofmap_new->index_map_bs() << std::endl;
  }
  assert(dofmap_new);

  // Build map from collapsed dof index to original dof index
  auto index_map_new = dofmap_new->index_map;
  const int map_bs = dofmap_new->index_map_bs();
  const std::int32_t size
      = map_bs * (index_map_new->size_local() + index_map_new->num_ghosts());
  std::vector<std::int32_t> collapsed_map(size);

  // if (this->bs() != dofmap_new->bs())
  if (this->bs() != 1)
    throw std::runtime_error("Block size problem");

  const int tdim = topology.dim();
  auto cells = topology.connectivity(tdim, 0);
  assert(cells);
  const int bs = dofmap_new->bs();
  std::cout << "Start loop" << std::endl;
  for (int c = 0; c < cells->num_nodes(); ++c)
  {
    auto cell_dofs_view = this->cell_dofs(c);
    auto cell_dofs = dofmap_new->cell_dofs(c);
    assert(cell_dofs_view.rows() == bs * cell_dofs.rows());
    assert(cell_dofs_view.rows() % bs == 0);
    for (Eigen::Index i = 0; i < cell_dofs.rows(); ++i)
    {
      for (int k = 0; k < bs; ++k)
      {
        assert(bs * cell_dofs[i] + k < (int)collapsed_map.size());
        assert(bs * i + k < cell_dofs_view.rows());
        collapsed_map[bs * cell_dofs[i] + k] = cell_dofs_view[bs * i + k];
      }
    }
    // for (Eigen::Index i = 0; i < cell_dofs_view.rows(); ++i)
    // {
    //   assert(cell_dofs[i] < (int)collapsed_map.size());
    //   collapsed_map[cell_dofs[i]] = cell_dofs_view[i];
    // }
  }
  // std::cout << "Step 5 " << std::endl;

  std::cout << "Done with collapse" << std::endl;
  return {std::move(dofmap_new), std::move(collapsed_map)};
}
//-----------------------------------------------------------------------------
const graph::AdjacencyList<std::int32_t>& DofMap::list() const
{
  return _dofmap;
}
//-----------------------------------------------------------------------------
int DofMap::bs() const { return _bs; }
//-----------------------------------------------------------------------------
int DofMap::index_map_bs() const { return _index_map_bs; }
//-----------------------------------------------------------------------------
