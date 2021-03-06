// Copyright (C) 2007-2020 Michal Habera, Anders Logg and Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <Eigen/Dense>
#include <array>
#include <dolfinx/function/Function.h>
#include <dolfinx/la/utils.h>
#include <functional>
#include <memory>
#include <vector>

namespace dolfinx
{

namespace function
{
template <typename T>
class Function;
class FunctionSpace;
} // namespace function

namespace mesh
{
class Mesh;
} // namespace mesh

namespace fem
{

/// Find degrees-of-freedom which belong to the provided mesh entities
/// (topological). Note that degrees-of-freedom for discontinuous
/// elements are associated with the cell even if they may appear to be
/// associated with a facet/edge/vertex.
///
/// @param[in] V The function (sub)spaces on which degrees-of-freedom
///   (DOFs) will be located. The spaces must share the same mesh and
///   element type.
/// @param[in] dim Topological dimension of mesh entities on which
///   degrees-of-freedom will be located
/// @param[in] entities Indices of mesh entities. All DOFs associated
///   with the closure of these indices will be returned
/// @param[in] remote True to return also "remotely located"
///   degree-of-freedom indices. Remotely located degree-of-freedom
///   indices are local/owned by the current process, but which the
///   current process cannot identify because it does not recognize mesh
///   entity as a marked. For example, a boundary condition dof at a
///   vertex where this process does not have the associated boundary
///   facet. This commonly occurs with partitioned meshes.
/// @return Array of local DOF indices in the spaces V[0] and V[1]. The
///   array[0](i) entry is the DOF index in the space V[0] and array[1](i)
///   is the correspinding DOF entry in the space V[1].
std::array<Eigen::Array<std::int32_t, Eigen::Dynamic, 1>, 2>
locate_dofs_topological(
    const std::array<std::reference_wrapper<const function::FunctionSpace>, 2>&
        V,
    const int dim, const Eigen::Ref<const Eigen::ArrayXi>& entities,
    bool remote = true);

/// Find degrees-of-freedom which belong to the provided mesh entities
/// (topological). Note that degrees-of-freedom for discontinuous
/// elements are associated with the cell even if they may appear to be
/// associated with a facet/edge/vertex.
///
/// @param[in] V The function (sub)space on which degrees-of-freedom
///   (DOFs) will be located.
/// @param[in] dim Topological dimension of mesh entities on which
///   degrees-of-freedom will be located
/// @param[in] entities Indices of mesh entities. All DOFs associated
///   with the closure of these indices will be returned
/// @param[in] remote True to return also "remotely located"
///   degree-of-freedom indices. Remotely located degree-of-freedom
///   indices are local/owned by the current process, but which the
///   current process cannot identify because it does not recognize mesh
///   entity as a marked. For example, a boundary condition dof at a
///   vertex where this process does not have the associated boundary
///   facet. This commonly occurs with partitioned meshes.
/// @return Array of local DOF indices in the spaces .
Eigen::Array<std::int32_t, Eigen::Dynamic, 1>
locate_dofs_topological(const function::FunctionSpace& V, const int dim,
                        const Eigen::Ref<const Eigen::ArrayXi>& entities,
                        bool remote = true);

/// Finds degrees of freedom whose geometric coordinate is true for the
/// provided marking function.
///
/// @attention This function is slower than the topological version
///
/// @param[in] V The function (sub)space(s) on which degrees of freedom
///     will be located. The spaces must share the same mesh and
///     element type.
/// @param[in] marker Function marking tabulated degrees of freedom
/// @return Array of local DOF indices in the spaces V[0] (and V[1] is
///     two spaces are passed in). If two spaces are passed in, the (i,
///     0) entry is the DOF index in the space V[0] and (i, 1) is the
///     correspinding DOF entry in the space V[1].
std::array<Eigen::Array<std::int32_t, Eigen::Dynamic, 1>, 2>
locate_dofs_geometrical(
    const std::array<std::reference_wrapper<const function::FunctionSpace>, 2>&
        V,
    const std::function<Eigen::Array<bool, Eigen::Dynamic, 1>(
        const Eigen::Ref<const Eigen::Array<double, 3, Eigen::Dynamic,
                                            Eigen::RowMajor>>&)>& marker);

/// Finds degrees of freedom whose geometric coordinate is true for the
/// provided marking function.
///
/// @attention This function is slower than the topological version
///
/// @param[in] V The function (sub)space on which degrees of freedom
///     will be located.
/// @param[in] marker Function marking tabulated degrees of freedom
/// @return Array of local DOF indices in the spaces.
Eigen::Array<std::int32_t, Eigen::Dynamic, 1> locate_dofs_geometrical(
    const function::FunctionSpace& V,
    const std::function<Eigen::Array<bool, Eigen::Dynamic, 1>(
        const Eigen::Ref<const Eigen::Array<double, 3, Eigen::Dynamic,
                                            Eigen::RowMajor>>&)>& marker);

/// Interface for setting (strong) Dirichlet boundary conditions
///
///     u = g on G,
///
/// where u is the solution to be computed, g is a function and G is a
/// sub domain of the mesh.
///
/// A DirichletBC is specified by the function g, the function space
/// (trial space) and degrees of freedom to which the boundary condition
/// applies.

template <typename T>
class DirichletBC
{

public:
  /// Create boundary condition
  ///
  /// @param[in] g The boundary condition value. The boundary condition
  /// can be applied to a a function on the same space as g.
  /// @param[in] dofs Degree-of-freedom indices in the space of the
  ///   boundary value function applied to V_dofs[i]
  DirichletBC(
      const std::shared_ptr<const function::Function<T>>& g,
      const Eigen::Ref<const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>>&
          dofs)
      : _function_space(g->function_space()), _g(g), _dofs({dofs, dofs})
  {
    // Stack indices as columns, fits column-major _dofs layout
    const int owned_size = _function_space->dofmap()->index_map_bs()
                           * _function_space->dofmap()->index_map->size_local();
    auto* it = std::lower_bound(_dofs[0].data(),
                                _dofs[0].data() + _dofs[0].rows(), owned_size);
    _owned_indices = std::distance(_dofs[0].data(), it);
  }

  /// Create boundary condition
  ///
  /// @param[in] g The boundary condition value
  /// @param[in] V_g_dofs Two arrays of degree-of-freedom indices. First
  ///   array are indices in the space where boundary condition is
  ///   applied (V), second array are indices in the space of the
  ///   boundary condition value function g.
  /// @param[in] V The function (sub)space on which the boundary
  ///   condition is applied
  DirichletBC(const std::shared_ptr<const function::Function<T>>& g,
              const std::array<Eigen::Array<std::int32_t, Eigen::Dynamic, 1>,
                               2>& V_g_dofs,
              std::shared_ptr<const function::FunctionSpace> V)
      : _function_space(V), _g(g), _dofs({V_g_dofs[0], V_g_dofs[1]})
  {
    const int owned_size = _function_space->dofmap()->index_map_bs()
                           * _function_space->dofmap()->index_map->size_local();
    auto* it = std::lower_bound(_dofs[0].data(),
                                _dofs[0].data() + _dofs[0].rows(), owned_size);
    _owned_indices = std::distance(_dofs[0].data(), it);
  }

  /// Copy constructor
  /// @param[in] bc The object to be copied
  DirichletBC(const DirichletBC& bc) = default;

  /// Move constructor
  /// @param[in] bc The object to be moved
  DirichletBC(DirichletBC&& bc) = default;

  /// Destructor
  ~DirichletBC() = default;

  /// Assignment operator
  /// @param[in] bc Another DirichletBC object
  DirichletBC& operator=(const DirichletBC& bc) = default;

  /// Move assignment operator
  DirichletBC& operator=(DirichletBC&& bc) = default;

  /// The function space to which boundary conditions are applied
  /// @return The function space
  std::shared_ptr<const function::FunctionSpace> function_space() const
  {
    return _function_space;
  }

  /// Return boundary value function g
  /// @return The boundary values Function
  std::shared_ptr<const function::Function<T>> value() const { return _g; }

  /// Get array of dof indices to which a Dirichlet boundary condition
  /// is applied. The array is sorted and may contain ghost entries.
  const std::array<Eigen::Array<std::int32_t, Eigen::Dynamic, 1>, 2>&
  dofs() const
  {
    return _dofs;
  }

  /// Get array of dof indices owned by this process to which a
  /// Dirichlet BC is applied. The array is sorted and does not contain
  /// ghost entries.
  const std::array<
      Eigen::Ref<const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>>, 2>
  dofs_owned() const
  {
    return {_dofs[0].head(_owned_indices), _dofs[1].head(_owned_indices)};
  }

  /// Set bc entries in x to scale*x_bc
  /// @todo Clarify w.r.t ghosts
  void set(Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, 1>> x,
           double scale = 1.0) const
  {
    // FIXME: This one excludes ghosts. Need to straighten out.
    assert(_g);
    auto& g = _g->x()->array();
    for (Eigen::Index i = 0; i < _dofs[0].rows(); ++i)
    {
      if (_dofs[0](i) < x.rows())
        x[_dofs[0](i)] = scale * g[_dofs[1](i)];
    }
  }

  /// Set bc entries in x to scale*(x0 - x_bc).
  /// @todo Clarify w.r.t ghosts
  void set(Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, 1>> x,
           const Eigen::Ref<const Eigen::Matrix<T, Eigen::Dynamic, 1>>& x0,
           double scale = 1.0) const
  {
    // FIXME: This one excludes ghosts. Need to straighten out.
    assert(_g);
    auto& g = _g->x()->array();
    assert(x.rows() <= x0.rows());
    for (Eigen::Index i = 0; i < _dofs[0].rows(); ++i)
    {
      if (_dofs[0](i) < x.rows())
        x[_dofs[0](i)] = scale * (g[_dofs[1](i)] - x0[_dofs[0](i)]);
    }
  }

  /// Set boundary condition value for entries with an applied boundary
  /// condition. Other entries are not modified.
  /// @todo Clarify w.r.t ghosts
  void dof_values(Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, 1>> values) const
  {
    assert(_g);
    auto& g = _g->x()->array();
    for (Eigen::Index i = 0; i < _dofs[0].rows(); ++i)
      values[_dofs[0](i)] = g[_dofs[1](i)];
  }

  /// Set markers[i] = true if dof i has a boundary condition applied.
  /// Value of markers[i] is not changed otherwise.
  /// @todo Clarify w.r.t ghosts
  void mark_dofs(std::vector<bool>& markers) const
  {
    for (Eigen::Index i = 0; i < _dofs[0].rows(); ++i)
    {
      assert(_dofs[0](i) < (std::int32_t)markers.size());
      markers[_dofs[0](i)] = true;
    }
  }

private:
  // The function space (possibly a sub function space)
  std::shared_ptr<const function::FunctionSpace> _function_space;

  // The function
  std::shared_ptr<const function::Function<T>> _g;

  // Pairs of dof indices in _function_space (i, 0) and in the space of
  // _g (i, 1)
  std::array<Eigen::Array<std::int32_t, Eigen::Dynamic, 1>, 2> _dofs;

  // The first _owned_indices in _dofs are owned by this process
  int _owned_indices = -1;
}; // namespace fem
} // namespace fem
} // namespace dolfinx
