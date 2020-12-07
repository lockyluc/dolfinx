// Copyright (C) 2020 Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "Partitioning.h"
#include "Mesh.h"
#include "Topology.h"
#include "cell_types.h"
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/Timer.h>
#include <dolfinx/common/log.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/graph/ParMETIS.h>
#include <dolfinx/graph/SCOTCH.h>
#include <dolfinx/mesh/GraphBuilder.h>

using namespace dolfinx;
using namespace dolfinx::mesh;

//-----------------------------------------------------------------------------
graph::AdjacencyList<std::int32_t> Partitioning::partition_cells(
    MPI_Comm comm, int nparts, const mesh::CellType cell_type,
    const graph::AdjacencyList<std::int64_t>& cells, mesh::GhostMode ghost_mode)
{
  common::Timer timer("Partition cells across processes");
  LOG(INFO) << "Compute partition of cells across processes";

  if (cells.num_nodes() > 0)
  {
    if (cells.num_links(0) != mesh::num_cell_vertices(cell_type))
    {
      throw std::runtime_error(
          "Inconsistent number of cell vertices. Got "
          + std::to_string(cells.num_links(0)) + ", expected "
          + std::to_string(mesh::num_cell_vertices(cell_type)) + ".");
    }
  }

  // Compute distributed dual graph (for the cells on this process)
  const auto [dual_graph, graph_info]
      = mesh::GraphBuilder::compute_dual_graph(comm, cells, cell_type);

  // Extract data from graph_info
  [[maybe_unused]] const auto [num_ghost_nodes, num_local_edges,
                               num_nonlocal_edges]
      = graph_info;

  std::vector<std::size_t> weights;

  // Just flag any kind of ghosting for now
  bool ghosting = (ghost_mode != mesh::GhostMode::none);

  const Partitioner partitioner = Partitioner::parmetis;

  switch (partitioner)
  {
  case Partitioner::scotch:
    return graph::SCOTCH::partition(
        comm, nparts, graph::AdjacencyList<SCOTCH_Num>(dual_graph), weights,
        num_ghost_nodes, ghosting);
  case Partitioner::parmetis:
#ifdef HAS_PARMETIS
    return graph::ParMETIS::partition(
        comm, nparts, graph::AdjacencyList<idx_t>(dual_graph), ghosting);
#else
    throw std::runtime_error("ParMETIS is not avaivalble.");
#endif
  default:
    throw std::runtime_error("Unknown graph partitioner.");
  }
}
//-----------------------------------------------------------------------------
