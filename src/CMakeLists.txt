set(HEADERS
  ExaMPM_BoundaryConditions.hpp
  ExaMPM_DenseLinearAlgebra.hpp
  ExaMPM_Mesh.hpp
  ExaMPM_ParticleCommunication.hpp
  ExaMPM_ParticleInit.hpp
  ExaMPM_ProblemManager.hpp
  ExaMPM_BATParticleWriter.hpp
  ExaMPM_Solver.hpp
  ExaMPM_TimeIntegrator.hpp
  ExaMPM_Types.hpp
  ExaMPM_VelocityInterpolation.hpp
  )

set(SOURCES
  ExaMPM_Mesh.cpp
  )

add_library(exampm ${SOURCES})

target_link_libraries(exampm
  Kokkos::kokkos
  MPI::MPI_CXX
  Cabana::cabanacore
  Cajita::Cajita
  bat_io
  TBB::tbb)

target_include_directories(exampm
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}
  )
