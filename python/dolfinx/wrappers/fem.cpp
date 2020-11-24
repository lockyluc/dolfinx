// Copyright (C) 2017-2019 Chris Richardson and Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "caster_mpi.h"
#include "caster_petsc.h"
#include <Eigen/Dense>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/types.h>
#include <dolfinx/fem/CoordinateElement.h>
#include <dolfinx/fem/DirichletBC.h>
#include <dolfinx/fem/DiscreteOperators.h>
#include <dolfinx/fem/DofMap.h>
#include <dolfinx/fem/DofMapBuilder.h>
#include <dolfinx/fem/ElementDofLayout.h>
#include <dolfinx/fem/FiniteElement.h>
#include <dolfinx/fem/Form.h>
#include <dolfinx/fem/assembler.h>
#include <dolfinx/fem/petsc.h>
#include <dolfinx/fem/utils.h>
#include <dolfinx/function/Constant.h>
#include <dolfinx/function/Expression.h>
#include <dolfinx/function/Function.h>
#include <dolfinx/function/FunctionSpace.h>
#include <dolfinx/la/PETScMatrix.h>
#include <dolfinx/la/PETScVector.h>
#include <dolfinx/la/SparsityPattern.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/MeshTags.h>
#include <memory>
#include <petsc4py/petsc4py.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <string>
#include <ufc.h>

namespace py = pybind11;

namespace
{
// Copy a vector-of-vectors into an Eigen::Array for dolfinx::fem::Form*
Eigen::Array<const dolfinx::fem::Form<PetscScalar>*, Eigen::Dynamic,
             Eigen::Dynamic, Eigen::RowMajor>
forms_vector_to_array(
    const std::vector<std::vector<const dolfinx::fem::Form<PetscScalar>*>>& a)
{
  if (a.empty())
  {
    return Eigen::Array<const dolfinx::fem::Form<PetscScalar>*, Eigen::Dynamic,
                        Eigen::Dynamic, Eigen::RowMajor>();
  }
  Eigen::Array<const dolfinx::fem::Form<PetscScalar>*, Eigen::Dynamic,
               Eigen::Dynamic, Eigen::RowMajor>
      _a(a.size(), a[0].size());
  _a = nullptr;
  for (std::size_t i = 0; i < a.size(); ++i)
  {
    if (a[i].size() != a[0].size())
      throw std::runtime_error("Array of forms is not rectangular.");
    for (std::size_t j = 0; j < a[i].size(); ++j)
      _a(i, j) = a[i][j];
  }
  return _a;
}

} // namespace

namespace dolfinx_wrappers
{
void fem(py::module& m)
{
  // utils
  m.def(
      "create_vector_block",
      [](const std::vector<
             std::reference_wrapper<const dolfinx::common::IndexMap>>& maps,
         const std::vector<int>& bs) {
        dolfinx::la::PETScVector x
            = dolfinx::fem::create_vector_block(maps, bs);
        Vec _x = x.vec();
        PetscObjectReference((PetscObject)_x);
        return _x;
      },
      py::return_value_policy::take_ownership,
      "Create a monolithic vector for multiple (stacked) linear forms.");
  m.def(
      "create_vector_nest",
      [](const std::vector<const dolfinx::common::IndexMap*>& maps,
         const std::vector<int>& bs) {
        auto x = dolfinx::fem::create_vector_nest(maps, bs);
        Vec _x = x.vec();
        PetscObjectReference((PetscObject)_x);
        return _x;
      },
      py::return_value_policy::take_ownership,
      "Create nested vector for multiple (stacked) linear forms.");

  m.def("create_sparsity_pattern",
        &dolfinx::fem::create_sparsity_pattern<PetscScalar>,
        "Create a sparsity pattern for bilinear form.");
  m.def("pack_coefficients",
        &dolfinx::fem::pack_coefficients<dolfinx::fem::Form<PetscScalar>>,
        "Pack coefficients for a UFL form.");
  m.def("pack_coefficients",
        &dolfinx::fem::pack_coefficients<dolfinx::fem::Form<PetscScalar>>,
        "Pack coefficients for a UFL expression.");
  m.def("pack_constants",
        &dolfinx::fem::pack_constants<dolfinx::fem::Form<PetscScalar>>,
        "Pack constants for a UFL form.");
  m.def(
      "pack_constants",
      &dolfinx::fem::pack_constants<dolfinx::function::Expression<PetscScalar>>,
      "Pack constants for a UFL expression.");
  m.def(
      "create_matrix",
      [](const dolfinx::fem::Form<PetscScalar>& a) {
        auto A = dolfinx::fem::create_matrix(a);
        Mat _A = A.mat();
        PetscObjectReference((PetscObject)_A);
        return _A;
      },
      py::return_value_policy::take_ownership,
      "Create a PETSc Mat for bilinear form.");
  m.def(
      "create_matrix_block",
      [](const std::vector<std::vector<const dolfinx::fem::Form<PetscScalar>*>>&
             a) {
        dolfinx::la::PETScMatrix A
            = dolfinx::fem::create_matrix_block(forms_vector_to_array(a));
        Mat _A = A.mat();
        PetscObjectReference((PetscObject)_A);
        return _A;
      },
      py::return_value_policy::take_ownership,
      "Create monolithic sparse matrix for stacked bilinear forms.");
  m.def(
      "create_matrix_nest",
      [](const std::vector<std::vector<const dolfinx::fem::Form<PetscScalar>*>>&
             a) {
        dolfinx::la::PETScMatrix A
            = dolfinx::fem::create_matrix_nest(forms_vector_to_array(a));
        Mat _A = A.mat();
        PetscObjectReference((PetscObject)_A);
        return _A;
      },
      py::return_value_policy::take_ownership,
      "Create nested sparse matrix for bilinear forms.");
  m.def(
      "create_element_dof_layout",
      [](const std::uintptr_t dofmap, const dolfinx::mesh::CellType cell_type,
         const std::vector<int>& parent_map) {
        const ufc_dofmap* p = reinterpret_cast<const ufc_dofmap*>(dofmap);
        return dolfinx::fem::create_element_dof_layout(*p, cell_type,
                                                       parent_map);
      },
      "Create ElementDofLayout object from a ufc dofmap.");
  m.def(
      "create_dofmap",
      [](const MPICommWrapper comm, const std::uintptr_t dofmap,
         dolfinx::mesh::Topology& topology) {
        const ufc_dofmap* p = reinterpret_cast<const ufc_dofmap*>(dofmap);
        return dolfinx::fem::create_dofmap(comm.get(), *p, topology);
      },
      "Create DofMap object from a pointer to ufc_dofmap.");
  m.def(
      "create_form",
      [](const std::uintptr_t form,
         const std::vector<
             std::shared_ptr<const dolfinx::function::FunctionSpace>>& spaces,
         const std::vector<std::shared_ptr<
             const dolfinx::function::Function<PetscScalar>>>& coefficients,
         const std::vector<std::shared_ptr<
             const dolfinx::function::Constant<PetscScalar>>>& constants,
         const std::map<dolfinx::fem::IntegralType,
                        const dolfinx::mesh::MeshTags<int>*>& subdomains,
         const std::shared_ptr<const dolfinx::mesh::Mesh>& mesh) {
        const ufc_form* p = reinterpret_cast<const ufc_form*>(form);
        return dolfinx::fem::create_form<PetscScalar>(
            *p, spaces, coefficients, constants, subdomains, mesh);
      },
      "Create Form from a pointer to ufc_form.");
  m.def(
      "create_coordinate_map",
      [](std::uintptr_t cmap) {
        const ufc_coordinate_mapping* p
            = reinterpret_cast<const ufc_coordinate_mapping*>(cmap);
        return dolfinx::fem::create_coordinate_map(*p);
      },
      "Create CoordinateElement from a pointer to ufc_coordinate_map.");
  m.def(
      "build_dofmap",
      [](const MPICommWrapper comm, const dolfinx::mesh::Topology& topology,
         const dolfinx::fem::ElementDofLayout& element_dof_layout) {
        auto [map, dofmap] = dolfinx::fem::DofMapBuilder::build(
            comm.get(), topology, element_dof_layout);
        return std::pair(map, std::move(dofmap));
      },
      "Build and dofmap on a mesh.");
  m.def("transpose_dofmap", &dolfinx::fem::transpose_dofmap,
        "Build the index to (cell, local index) map from a "
        "dofmap ((cell, local index ) -> index).");

  // dolfinx::fem::FiniteElement
  py::class_<dolfinx::fem::FiniteElement,
             std::shared_ptr<dolfinx::fem::FiniteElement>>(
      m, "FiniteElement", "Finite element object")
      .def(py::init([](const std::uintptr_t ufc_element) {
        const ufc_finite_element* p
            = reinterpret_cast<const ufc_finite_element*>(ufc_element);
        return dolfinx::fem::FiniteElement(*p);
      }))
      .def("num_sub_elements", &dolfinx::fem::FiniteElement::num_sub_elements)
      .def("dof_reference_coordinates",
           &dolfinx::fem::FiniteElement::dof_reference_coordinates)
      .def_property_readonly("value_rank",
                             &dolfinx::fem::FiniteElement::value_rank)
      .def_property_readonly("block_size",
                             &dolfinx::fem::FiniteElement::block_size)
      .def("space_dimension", &dolfinx::fem::FiniteElement::space_dimension)
      .def("value_dimension", &dolfinx::fem::FiniteElement::value_dimension)
      .def("signature", &dolfinx::fem::FiniteElement::signature);

  // dolfinx::fem::ElementDofLayout
  py::class_<dolfinx::fem::ElementDofLayout,
             std::shared_ptr<dolfinx::fem::ElementDofLayout>>(
      m, "ElementDofLayout", "Object describing the layout of dofs on a cell")
      .def(py::init<int, const std::vector<std::vector<std::set<int>>>&,
                    const std::vector<int>&,
                    const std::vector<
                        std::shared_ptr<const dolfinx::fem::ElementDofLayout>>,
                    const dolfinx::mesh::CellType,
                    const Eigen::Array<int, Eigen::Dynamic, Eigen::Dynamic,
                                       Eigen::RowMajor>&>())
      .def_property_readonly("num_dofs",
                             &dolfinx::fem::ElementDofLayout::num_dofs)
      .def("num_entity_dofs", &dolfinx::fem::ElementDofLayout::num_entity_dofs)
      .def("num_entity_closure_dofs",
           &dolfinx::fem::ElementDofLayout::num_entity_closure_dofs)
      .def("entity_dofs", &dolfinx::fem::ElementDofLayout::entity_dofs)
      .def("entity_closure_dofs",
           &dolfinx::fem::ElementDofLayout::entity_closure_dofs)
      .def("block_size", &dolfinx::fem::ElementDofLayout::block_size);

  // dolfinx::fem::DofMap
  py::class_<dolfinx::fem::DofMap, std::shared_ptr<dolfinx::fem::DofMap>>(
      m, "DofMap", "DofMap object")
      .def(py::init<std::shared_ptr<const dolfinx::fem::ElementDofLayout>,
                    std::shared_ptr<const dolfinx::common::IndexMap>, int,
                    dolfinx::graph::AdjacencyList<std::int32_t>&, int>(),
           py::arg("element_dof_layout"), py::arg("index_map"),
           py::arg("index_map_bs"), py::arg("dofmap"), py::arg("bs"))
      .def_readonly("index_map", &dolfinx::fem::DofMap::index_map)
      .def_property_readonly("index_map_bs",
                             &dolfinx::fem::DofMap::index_map_bs)
      .def_property_readonly("bs", &dolfinx::fem::DofMap::bs)
      .def_readonly("dof_layout", &dolfinx::fem::DofMap::element_dof_layout)
      .def("cell_dofs", &dolfinx::fem::DofMap::cell_dofs)
      .def("list", &dolfinx::fem::DofMap::list);

  // dolfinx::fem::CoordinateElement
  py::class_<dolfinx::fem::CoordinateElement,
             std::shared_ptr<dolfinx::fem::CoordinateElement>>(
      m, "CoordinateElement", "Coordinate map element")
      .def_property_readonly("dof_layout",
                             &dolfinx::fem::CoordinateElement::dof_layout)
      .def("push_forward", &dolfinx::fem::CoordinateElement::push_forward);

  // dolfinx::fem::DirichletBC
  py::class_<dolfinx::fem::DirichletBC<PetscScalar>,
             std::shared_ptr<dolfinx::fem::DirichletBC<PetscScalar>>>
      dirichletbc(
          m, "DirichletBC",
          "Object for representing Dirichlet (essential) boundary conditions");

  dirichletbc
      .def(py::init<
               std::shared_ptr<const dolfinx::function::Function<PetscScalar>>,
               const Eigen::Ref<
                   const Eigen::Array<std::int32_t, Eigen::Dynamic, 2>>&,
               std::shared_ptr<const dolfinx::function::FunctionSpace>>(),
           py::arg("V"), py::arg("g"), py::arg("V_g_dofs"))
      .def(py::init<
               std::shared_ptr<const dolfinx::function::Function<PetscScalar>>,
               const Eigen::Ref<
                   const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>>&>(),
           py::arg("g"), py::arg("dofs"))
      .def_property_readonly("dof_indices",
                             &dolfinx::fem::DirichletBC<PetscScalar>::dofs)
      .def_property_readonly(
          "function_space",
          &dolfinx::fem::DirichletBC<PetscScalar>::function_space)
      .def_property_readonly("value",
                             &dolfinx::fem::DirichletBC<PetscScalar>::value);

  // dolfinx::fem::assemble
  // Functional
  m.def("assemble_scalar", &dolfinx::fem::assemble_scalar<PetscScalar>,
        "Assemble functional over mesh");
  // Vector
  m.def("assemble_vector", &dolfinx::fem::assemble_vector<PetscScalar>,
        py::arg("b"), py::arg("L"),
        "Assemble linear form into an existing Eigen vector");
  // Matrices
  m.def("assemble_matrix_petsc",
        [](Mat A, const dolfinx::fem::Form<PetscScalar>& a,
           const std::vector<std::shared_ptr<
               const dolfinx::fem::DirichletBC<PetscScalar>>>& bcs) {
          //   dolfinx::fem::assemble_matrix(dolfinx::la::PETScMatrix::add_fn(A),
          //   a,
          //                                 bcs);
          dolfinx::fem::assemble_matrix(
              dolfinx::la::PETScMatrix::add_fn_block(A), a, bcs);
        });
  m.def("assemble_matrix_petsc",
        [](Mat A, const dolfinx::fem::Form<PetscScalar>& a,
           const std::vector<bool>& rows0, const std::vector<bool>& rows1) {
          dolfinx::fem::assemble_matrix(dolfinx::la::PETScMatrix::add_fn(A), a,
                                        rows0, rows1);
        });
  m.def("add_diagonal",
        [](Mat A, const dolfinx::function::FunctionSpace& V,
           const std::vector<std::shared_ptr<
               const dolfinx::fem::DirichletBC<PetscScalar>>>& bcs,
           PetscScalar diagonal) {
          dolfinx::fem::add_diagonal(dolfinx::la::PETScMatrix::add_fn(A), V,
                                     bcs, diagonal);
        });
  m.def("assemble_matrix",
        py::overload_cast<const std::function<int(
                              std::int32_t, const std::int32_t*, std::int32_t,
                              const std::int32_t*, const PetscScalar*)>&,
                          const dolfinx::fem::Form<PetscScalar>&,
                          const std::vector<std::shared_ptr<
                              const dolfinx::fem::DirichletBC<PetscScalar>>>&>(
            &dolfinx::fem::assemble_matrix<PetscScalar>));

  // BC modifiers
  m.def("apply_lifting", &dolfinx::fem::apply_lifting<PetscScalar>,
        "Modify vector for lifted boundary conditions");
  m.def(
      "set_bc",
      [](Eigen::Ref<Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>> b,
         const std::vector<std::shared_ptr<
             const dolfinx::fem::DirichletBC<PetscScalar>>>& bcs,
         const py::array_t<PetscScalar>& x0, double scale) {
        if (x0.ndim() == 0)
          dolfinx::fem::set_bc<PetscScalar>(b, bcs, scale);
        else if (x0.ndim() == 1)
        {
          Eigen::Map<const Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>> _x0(
              x0.data(), x0.shape(0));
          dolfinx::fem::set_bc<PetscScalar>(b, bcs, _x0, scale);
        }
        else
          throw std::runtime_error("Wrong array dimension.");
      },
      py::arg("b"), py::arg("bcs"), py::arg("x0") = py::none(),
      py::arg("scale") = 1.0);
  // Tools
  m.def("bcs_rows", &dolfinx::fem::bcs_rows<PetscScalar>);
  m.def("bcs_cols", &dolfinx::fem::bcs_cols<PetscScalar>);

  //   // dolfinx::fem::DiscreteOperators
  //   py::class_<dolfinx::fem::DiscreteOperators>(m, "DiscreteOperators")
  //       .def_static(
  //           "build_gradient",
  //           [](const dolfinx::function::FunctionSpace& V0,
  //              const dolfinx::function::FunctionSpace& V1) {
  //             dolfinx::la::PETScMatrix A
  //                 = dolfinx::fem::DiscreteOperators::build_gradient(V0, V1);
  //             Mat _A = A.mat();
  //             PetscObjectReference((PetscObject)_A);
  //             return _A;
  //           },
  //           py::return_value_policy::take_ownership);

  py::enum_<dolfinx::fem::IntegralType>(m, "IntegralType")
      .value("cell", dolfinx::fem::IntegralType::cell)
      .value("exterior_facet", dolfinx::fem::IntegralType::exterior_facet)
      .value("interior_facet", dolfinx::fem::IntegralType::interior_facet)
      .value("vertex", dolfinx::fem::IntegralType::vertex);

  // dolfinx::fem::Form
  py::class_<dolfinx::fem::Form<PetscScalar>,
             std::shared_ptr<dolfinx::fem::Form<PetscScalar>>>(
      m, "Form", "Variational form object")
      .def(py::init([](const std::vector<std::shared_ptr<
                           const dolfinx::function::FunctionSpace>>& spaces,
                       const std::map<
                           dolfinx::fem::IntegralType,
                           std::pair<std::vector<std::pair<int, py::object>>,
                                     const dolfinx::mesh::MeshTags<int>*>>&
                           integrals,
                       const std::vector<std::shared_ptr<
                           const dolfinx::function::Function<PetscScalar>>>&
                           coefficients,
                       const std::vector<std::shared_ptr<
                           const dolfinx::function::Constant<PetscScalar>>>&
                           constants,
                       bool needs_permutation_data,
                       const std::shared_ptr<const dolfinx::mesh::Mesh>& mesh) {
             using kern = std::function<void(PetscScalar*, const PetscScalar*,
                                             const PetscScalar*, const double*,
                                             const int*, const std::uint8_t*,
                                             const std::uint32_t)>;
             std::map<dolfinx::fem::IntegralType,
                      std::pair<std::vector<std::pair<int, kern>>,
                                const dolfinx::mesh::MeshTags<int>*>>
                 _integrals;

             // Loop over kernel for each entity type
             for (auto& kernel_type : integrals)
             {
               // Set subdomain markers
               _integrals[kernel_type.first].second = nullptr;

               // Loop over each domain kernel
               for (auto& kernel : kernel_type.second.first)
               {
                 auto tabulate_tensor_ptr = (void (*)(
                     PetscScalar*, const PetscScalar*, const PetscScalar*,
                     const double*, const int*, const std::uint8_t*,
                     const std::uint32_t))kernel.second.cast<std::uintptr_t>();
                 _integrals[kernel_type.first].first.push_back(
                     {kernel.first, tabulate_tensor_ptr});
               }
             }
             return dolfinx::fem::Form<PetscScalar>(
                 spaces, _integrals, coefficients, constants,
                 needs_permutation_data, mesh);
           }),
           py::arg("spaces"), py::arg("integrals"), py::arg("coefficients"),
           py::arg("constants"), py::arg("need_permutation_data"),
           py::arg("mesh") = py::none())
      .def_property_readonly("coefficients",
                             &dolfinx::fem::Form<PetscScalar>::coefficients)
      .def_property_readonly("rank", &dolfinx::fem::Form<PetscScalar>::rank)
      .def_property_readonly("mesh", &dolfinx::fem::Form<PetscScalar>::mesh)
      .def_property_readonly("function_spaces",
                             &dolfinx::fem::Form<PetscScalar>::function_spaces)
      .def("integral_ids", &dolfinx::fem::Form<PetscScalar>::integral_ids)
      .def("domains", &dolfinx::fem::Form<PetscScalar>::domains);

  m.def("locate_dofs_topological", &dolfinx::fem::locate_dofs_topological,
        py::arg("V"), py::arg("dim"), py::arg("entities"),
        py::arg("remote") = true);
  m.def("locate_dofs_geometrical", &dolfinx::fem::locate_dofs_geometrical);
}
} // namespace dolfinx_wrappers
