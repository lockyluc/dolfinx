// Copyright (C) 2020 Igor Baratta
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <dolfinx/graph/AdjacencyList.h>

namespace dolfinx::graph
{
template <typename T>
graph::AdjacencyList<std::int32_t>
transpose(const graph::AdjacencyList<T>& adj_list)
{
  const std::int32_t num_nodes = adj_list.num_nodes();
  const std::int32_t num_vertices = adj_list.array().size() + 1;
  const std::size_t num_connections = adj_list.array().size();

  Eigen::Array<std::int32_t, Eigen::Dynamic, 1> array(num_connections);
  Eigen::Array<std::int32_t, Eigen::Dynamic, 1> offsets(num_vertices + 1);

  std::vector<std::int32_t> count(num_vertices);
  auto& connections = adj_list.array();
  for (std::size_t i = 0; i < num_connections; i++)
    count[std::size_t(connections[i])]++;

  std::exclusive_scan(count.begin(), count.end(), offsets.data(), 0);
  offsets[num_vertices] = num_connections;

  std::fill(count.begin(), count.end(), 0);

  for (std::int32_t i = 0; i < num_nodes; i++)
  {
    const auto& verts = adj_list.links(i);
    int num_verts = verts.size();
    for (int j = 0; j < num_verts; ++j)
    {
      auto vert = static_cast<std::int32_t>(verts[j]);
      array[offsets[vert] + count[vert]] = i;
      count[vert]++;
    }
  }

  return graph::AdjacencyList<std::int32_t>(array, offsets);
}
} // namespace dolfinx::graph