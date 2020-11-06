// Copyright (C) 2010-2013 Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "GraphBuilder.h"

#include <algorithm>
#include <boost/unordered_map.hpp>
#include <set>
#include <unordered_set>
#include <utility>
#include <vector>

#include <dolfinx/common/MPI.h>
#include <dolfinx/common/Timer.h>
#include <dolfinx/common/log.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/graph/utils.h>
#include <dolfinx/mesh/cell_types.h>

using namespace dolfinx;

namespace
{

[[maybe_unused]] std::vector<std::vector<std::int32_t>>
compute_local_dual_graph_impl(
    const graph::AdjacencyList<std::int64_t>& cell_vertices,
    const mesh::CellType& cell_type, const int sharing_dim)
{
  const int tdim = mesh::cell_dim(cell_type);

  if (sharing_dim < 0 || sharing_dim >= tdim)
    throw std::runtime_error("Sharing dim should be less");

  common::Timer timer("Compute local part of mesh dual graph -  proposal");

  const std::int32_t num_local_cells = cell_vertices.num_nodes();
  const std::int32_t num_local_vertices = cell_vertices.array().maxCoeff() + 1;

  // Number of vertices that have to be shared so an edge (in the dual graph) is
  // estabilished between two cells
  const int num_vertices_per_entity
      = mesh::num_cell_vertices(mesh::cell_entity_type(cell_type, sharing_dim));

  auto vert_cells = dolfinx::graph::transpose<std::int64_t>(cell_vertices);

  std::vector<int32_t> array;
  std::vector<int32_t> cell_count(num_local_cells);
  std::vector<int32_t> offsets(num_local_cells + 1);

  for (std::int32_t i = 0; i < num_local_vertices; i++)
  {
    const auto& cells = vert_cells.links(i);
    int num_cells_vert = cells.size();
    for (std::int32_t j = 0; j < num_cells_vert; j++)
      cell_count[cells[j]] += num_cells_vert;
  }

  offsets.resize(num_local_cells + 1);
  std::exclusive_scan(cell_count.begin(), cell_count.end(), offsets.begin(), 0);
  offsets[num_local_cells] = cell_count.back() + offsets[num_local_cells - 1];

  array.resize(offsets.back());
  std::fill(cell_count.begin(), cell_count.end(), 0);

  for (std::int32_t i = 0; i < num_local_vertices; i++)
  {
    const auto& cells = vert_cells.links(i);
    int num_cells_vert = cells.size();
    for (std::int32_t j = 0; j < num_cells_vert; j++)
    {
      std::int32_t offset = offsets[cells[j]];
      auto& local_pos = cell_count[cells[j]];
      for (std::int32_t k = 0; k < num_cells_vert; k++)
        array[offset + local_pos++] = cells[k];
    }
  }

  std::vector<std::vector<std::int32_t>> local_graph(num_local_cells);
  for (std::int32_t i = 0; i < num_local_cells; i++)
  {
    auto cell_begin = array.begin() + offsets[i];
    auto cell_end = array.begin() + offsets[i + 1];

    std::sort(cell_begin, cell_end);

    auto iter = cell_begin;
    while (iter != cell_end)
    {
      auto next = std::upper_bound(iter, cell_end, *iter);
      int rep = std::distance(iter, next);
      if (rep >= num_vertices_per_entity && *iter != i)
        local_graph[i].push_back(*iter);
      iter = next;
    }
  }

  for (auto cell_cell : local_graph)
  {
    std::cout << std::endl;
    for (auto cell : cell_cell)
      std::cout << cell << " ";
  }
  std::cout << std::endl;

  return local_graph;
}

//-----------------------------------------------------------------------------
// Compute local part of the dual graph, and return return (local_graph,
// facet_cell_map, number of local edges in the graph (undirected)
template <int N>
std::tuple<std::vector<std::vector<std::int32_t>>,
           std::vector<std::pair<std::vector<std::int32_t>, std::int32_t>>,
           std::int32_t>
compute_local_dual_graph_keyed(
    const graph::AdjacencyList<std::int64_t>& cell_vertices,
    const mesh::CellType& cell_type)
{
  common::Timer timer("Compute local part of mesh dual graph");

  const int tdim = mesh::cell_dim(cell_type);
  const std::int32_t num_local_cells = cell_vertices.num_nodes();
  const int num_facets_per_cell = mesh::cell_num_entities(cell_type, tdim - 1);
  const int num_vertices_per_facet
      = mesh::num_cell_vertices(mesh::cell_entity_type(cell_type, tdim - 1));

  assert(N == num_vertices_per_facet);

  // Compute edges (cell-cell connections) using local numbering

  // Create map from cell vertices to entity vertices
  const Eigen::Array<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      facet_vertices = mesh::get_entity_vertices(cell_type, tdim - 1);

  // Vector-of-arrays data structure, which is considerably faster than
  // vector-of-vectors
  std::vector<std::pair<std::array<std::int32_t, N>, std::int32_t>> facets(
      num_facets_per_cell * num_local_cells);

  // Iterate over all cells and build list of all facets (keyed on
  // sorted vertex indices), with cell index attached
  int counter = 0;
  for (std::int32_t i = 0; i < num_local_cells; ++i)
  {
    auto& vertices = cell_vertices.links(i);
    // Iterate over facets of cell
    for (int j = 0; j < num_facets_per_cell; ++j)
    {
      // Get list of facet vertices
      auto& facet = facets[counter].first;
      for (int k = 0; k < N; ++k)
        facet[k] = vertices[facet_vertices(j, k)];

      // Sort facet vertices
      std::sort(facet.begin(), facet.end());

      // Attach local cell index
      facets[counter].second = i;

      // Increment facet counter
      counter++;
    }
  }

  // Sort facets
  std::sort(facets.begin(), facets.end());

  // Find maching facets by comparing facet i and facet i -1
  std::size_t num_local_edges = 0;
  std::vector<std::vector<std::int32_t>> local_graph(num_local_cells);
  std::vector<std::pair<std::vector<std::int32_t>, std::int32_t>>
      facet_cell_map;
  for (std::size_t i = 1; i < facets.size(); ++i)
  {
    const int ii = i;
    const int jj = i - 1;

    const auto& facet0 = facets[jj].first;
    const auto& facet1 = facets[ii].first;
    const int cell_index0 = facets[jj].second;
    if (std::equal(facet1.begin(), facet1.end(), facet0.begin()))
    {
      // Add edges (directed graph, so add both ways)
      const int cell_index1 = facets[ii].second;
      local_graph[cell_index0].push_back(cell_index1);
      local_graph[cell_index1].push_back(cell_index0);

      // Since we've just found a matching pair, the next pair cannot be
      // matching, so advance 1
      ++i;

      // Increment number of local edges found
      ++num_local_edges;
    }
    else
    {
      // No match, so add facet0 to map
      facet_cell_map.emplace_back(
          std::vector<std::int32_t>(facet0.begin(), facet0.end()), cell_index0);
    }
  }

  // Add last facet, as it's not covered by the above loop. We could
  // check it against the preceding facet, but it's easier to just
  // insert it here
  if (!facets.empty())
  {
    const int k = facets.size() - 1;
    const int cell_index = facets[k].second;
    facet_cell_map.emplace_back(
        std::vector<std::int32_t>(facets[k].first.begin(),
                                  facets[k].first.end()),
        cell_index);
  }

  // for (auto cell_cell : local_graph)
  // {
  //   std::cout << std::endl;
  //   for (auto cell : cell_cell)
  //     std::cout << cell << " ";
  // }

  return {std::move(local_graph), std::move(facet_cell_map), num_local_edges};
}
//-----------------------------------------------------------------------------
// Build nonlocal part of dual graph for mesh and return number of
// non-local edges. Note: GraphBuilder::compute_local_dual_graph should
// be called before this function is called. Returns (ghost vertices,
// num_nonlocal_edges)
std::tuple<std::vector<std::vector<std::int64_t>>, std::int32_t, std::int32_t>
compute_nonlocal_dual_graph(
    const MPI_Comm mpi_comm,
    const graph::AdjacencyList<std::int64_t>& cell_vertices,
    const mesh::CellType& cell_type,
    const std::vector<std::pair<std::vector<std::int32_t>, std::int32_t>>&
        facet_cell_map,
    const std::vector<std::vector<std::int32_t>>& local_graph)
{
  LOG(INFO) << "Build nonlocal part of mesh dual graph";
  common::Timer timer("Compute non-local part of mesh dual graph");

  const std::int32_t num_local_cells = cell_vertices.num_nodes();

  // Get offset for this process
  const std::int64_t offset
      = dolfinx::MPI::global_offset(mpi_comm, num_local_cells, true);

  std::vector<std::vector<std::int64_t>> graph(local_graph.size());
  for (std::size_t i = 0; i < local_graph.size(); ++i)
  {
    graph[i] = std::vector<std::int64_t>(local_graph[i].begin(),
                                         local_graph[i].end());
    std::for_each(graph[i].begin(), graph[i].end(),
                  [offset](auto& n) { n += offset; });
  }

  // Get number of MPI processes, and return if mesh is not distributed
  const int num_processes = dolfinx::MPI::size(mpi_comm);
  if (num_processes == 1)
    return {graph, 0, 0};

  // At this stage facet_cell map only contains facets->cells with edge
  // facets either interprocess or external boundaries

  const int tdim = mesh::cell_dim(cell_type);

  // List of cell vertices
  const int num_vertices_per_facet
      = mesh::num_cell_vertices(mesh::cell_entity_type(cell_type, tdim - 1));

  //  assert(num_vertices_per_cell == (int)cell_vertices.cols());

  // Compute local edges (cell-cell connections) using global (internal
  // to this function, not the user numbering) numbering

  // Get global range of vertex indices
  std::int64_t num_global_vertices = 0;
  const std::int64_t max_vertex
      = (cell_vertices.num_nodes() > 0) ? cell_vertices.array().maxCoeff() : 0;
  MPI_Allreduce(&max_vertex, &num_global_vertices, 1, MPI_INT64_T, MPI_SUM,
                mpi_comm);
  num_global_vertices += 1;

  // Send facet-cell map to intermediary match-making processes
  std::vector<std::vector<std::int64_t>> send_buffer(num_processes);

  // Pack map data and send to match-maker process
  for (const auto& it : facet_cell_map)
  {
    // FIXME: Could use a better index? First vertex is slightly
    //        skewed towards low values - may not be important

    // Use first vertex of facet to partition into blocks
    const int dest_proc = dolfinx::MPI::index_owner(
        num_processes, (it.first)[0], num_global_vertices);

    // Pack map into vectors to send
    send_buffer[dest_proc].insert(send_buffer[dest_proc].end(),
                                  it.first.begin(), it.first.end());

    // Add offset to cell numbers sent off process
    send_buffer[dest_proc].push_back(it.second + offset);
  }

  // Send data
  const graph::AdjacencyList<std::int64_t> received_buffer
      = dolfinx::MPI::all_to_all(
          mpi_comm, graph::AdjacencyList<std::int64_t>(send_buffer));

  // Clear send buffer
  send_buffer = std::vector<std::vector<std::int64_t>>(num_processes);

  // Map to connect processes and cells, using facet as key
  typedef boost::unordered_map<std::vector<std::int64_t>,
                               std::pair<std::int64_t, std::int64_t>>
      MatchMap;
  MatchMap matchmap;

  // Look for matches to send back to other processes
  std::pair<std::vector<std::int64_t>, std::pair<std::int64_t, std::int64_t>>
      key;
  key.first.resize(num_vertices_per_facet);
  for (int p = 0; p < num_processes; ++p)
  {
    // Unpack into map
    auto data_p = received_buffer.links(p);
    for (int i = 0; i < data_p.rows(); i += (num_vertices_per_facet + 1))
    {
      // Build map key
      std::copy(data_p.data() + i, data_p.data() + i + num_vertices_per_facet,
                key.first.begin());
      key.second.first = p;
      key.second.second = data_p[i + num_vertices_per_facet];

      // Perform map insertion/look-up
      std::pair<MatchMap::iterator, bool> data = matchmap.insert(key);

      // If data is already in the map, extract data and remove from map
      if (!data.second)
      {
        // Found a match of two facets - send back to owners
        const std::size_t proc1 = data.first->second.first;
        const std::size_t proc2 = p;
        const std::size_t cell1 = data.first->second.second;
        const std::size_t cell2 = key.second.second;
        send_buffer[proc1].push_back(cell1);
        send_buffer[proc1].push_back(cell2);
        send_buffer[proc2].push_back(cell2);
        send_buffer[proc2].push_back(cell1);

        // Remove facet - saves memory and search time
        matchmap.erase(data.first);
      }
    }
  }

  // Send matches to other processes
  const Eigen::Array<std::int64_t, Eigen::Dynamic, 1> cell_list
      = dolfinx::MPI::all_to_all(
            mpi_comm, graph::AdjacencyList<std::int64_t>(send_buffer))
            .array();

  // Ghost nodes: insert connected cells into local map
  std::set<std::int64_t> ghost_nodes;
  std::int32_t num_nonlocal_edges = 0;
  for (int i = 0; i < cell_list.rows(); i += 2)
  {
    assert((std::int64_t)cell_list[i] >= offset);
    assert((std::int64_t)(cell_list[i] - offset)
           < (std::int64_t)local_graph.size());

    auto& edges = graph[cell_list[i] - offset];
    auto it = std::find(edges.begin(), edges.end(), cell_list[i + 1]);
    if (it == graph[cell_list[i] - offset].end())
    {
      edges.push_back(cell_list[i + 1]);
      ++num_nonlocal_edges;
    }
    ghost_nodes.insert(cell_list[i + 1]);
  }

  return {std::move(graph), ghost_nodes.size(), num_nonlocal_edges};
}
//-----------------------------------------------------------------------------

} // namespace

//-----------------------------------------------------------------------------
std::pair<std::vector<std::vector<std::int64_t>>, std::array<std::int32_t, 3>>
mesh::GraphBuilder::compute_dual_graph(
    const MPI_Comm mpi_comm,
    const graph::AdjacencyList<std::int64_t>& cell_vertices,
    const mesh::CellType& cell_type)
{
  LOG(INFO) << "Build mesh dual graph";

  // Compute local part of dual graph
  auto [local_graph, facet_cell_map, num_local_edges]
      = mesh::GraphBuilder::compute_local_dual_graph(cell_vertices, cell_type);

  // Compute nonlocal part
  auto [graph, num_ghost_nodes, num_nonlocal_edges]
      = compute_nonlocal_dual_graph(mpi_comm, cell_vertices, cell_type,
                                    facet_cell_map, local_graph);

  return {std::move(graph),
          {num_ghost_nodes, num_local_edges, num_nonlocal_edges}};
}
//-----------------------------------------------------------------------------
std::tuple<std::vector<std::vector<std::int32_t>>,
           std::vector<std::pair<std::vector<std::int32_t>, std::int32_t>>,
           std::int32_t>
dolfinx::mesh::GraphBuilder::compute_local_dual_graph(
    const graph::AdjacencyList<std::int64_t>& cell_vertices,
    const mesh::CellType& cell_type)
{
  LOG(INFO) << "Build local part of mesh dual graph";

  const int tdim = mesh::cell_dim(cell_type);
  const int num_entity_vertices
      = mesh::num_cell_vertices(mesh::cell_entity_type(cell_type, tdim - 1));

  compute_local_dual_graph_impl(cell_vertices, cell_type, tdim - 1);

  switch (num_entity_vertices)
  {
  case 1:
    return compute_local_dual_graph_keyed<1>(cell_vertices, cell_type);
  case 2:
    return compute_local_dual_graph_keyed<2>(cell_vertices, cell_type);
  case 3:
    return compute_local_dual_graph_keyed<3>(cell_vertices, cell_type);
  case 4:
    return compute_local_dual_graph_keyed<4>(cell_vertices, cell_type);
  default:
    throw std::runtime_error(
        "Cannot compute local part of dual graph. Entities with "
        + std::to_string(num_entity_vertices) + " vertices not supported");
  }
}
//-----------------------------------------------------------------------------
