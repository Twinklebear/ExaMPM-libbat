/****************************************************************************
 * Copyright (c) 2018-2020 by the ExaMPM authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the ExaMPM library. ExaMPM is distributed under a   *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#ifndef EXAMPM_BATPARTICLEWRITER_HPP
#define EXAMPM_BATPARTICLEWRITER_HPP

#include <unistd.h>
#include <cstring>
#include <fstream>
#include <memory>
#include <cmath>
#include <chrono>

#include <Cajita.hpp>

#include <Cabana_Core.hpp>

#include <Kokkos_Core.hpp>

#include <bat_io.h>

#include <mpi.h>

#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

namespace ExaMPM
{
namespace BATParticleWriter
{
//---------------------------------------------------------------------------//
// BAT Particle Field Writer.
//---------------------------------------------------------------------------//
// Format traits.
template<typename T>
struct BATTraits;

template<>
struct BATTraits<short>
{
    static BATDataType type(int n_comp)
    {
        switch (n_comp) {
            case 1: return BTD_INT_16;
            case 2: return BTD_VEC2_I16;
            case 3: return BTD_VEC3_I16;
            case 4: return BTD_VEC4_I16;
            default: BTD_UNKNOWN;
        }
    }

    static int stride(int n_comp)
    {
        switch (n_comp) {
            case 1: return 2;
            case 2: return 4;
            case 3: return 6;
            case 4: return 8;
            default: return -1;
        }
    }
};

template<>
struct BATTraits<int>
{
    static BATDataType type(int n_comp)
    {
        switch (n_comp) {
            case 1: return BTD_INT_32;
            case 2: return BTD_VEC2_I32;
            case 3: return BTD_VEC3_I32;
            case 4: return BTD_VEC4_I32;
            default: BTD_UNKNOWN;
        }
    }

    static int stride(int n_comp)
    {
        switch (n_comp) {
            case 1: return 4;
            case 2: return 8;
            case 3: return 12;
            case 4: return 16;
            default: return -1;
        }
    }
};

template<>
struct BATTraits<float>
{
    static BATDataType type(int n_comp)
    {
        switch (n_comp) {
            case 1: return BTD_FLOAT_32;
            case 2: return BTD_VEC2_FLOAT;
            case 3: return BTD_VEC3_FLOAT;
            case 4: return BTD_VEC4_FLOAT;
            default: return BTD_UNKNOWN;
        }
    }

    static int stride(int n_comp)
    {
        switch (n_comp) {
            case 1: return 4;
            case 2: return 8;
            case 3: return 12;
            case 4: return 16;
            default: return -1;
        }
    }
};

template<>
struct BATTraits<double>
{
    static BATDataType type(int n_comp)
    {
        switch (n_comp) {
            case 1: return BTD_FLOAT_64;
            case 2: return BTD_VEC2_DOUBLE;
            case 3: return BTD_VEC3_DOUBLE;
            case 4: return BTD_VEC4_DOUBLE;
            default: return BTD_UNKNOWN;
        }
    }

    static int stride(int n_comp)
    {
        switch (n_comp) {
            case 1: return 8;
            case 2: return 16;
            case 3: return 24;
            case 4: return 32;
            default: return -1;

        }
    }
};

// TODO: A direct AoSoA API for libbat. For now we need keep
// the linear arrays alive until the actual write is called
// b/c libbat doesn't buffer internally
using DataHandle = std::shared_ptr<uint8_t>;
struct BATData {
    std::vector<DataHandle> data;
};

//---------------------------------------------------------------------------//
// Rank-0 field
template<class SliceType>
void writeFieldsImpl(
    BATParticleState bat_file,
    BATData &data,
    const std::string& mesh_name,
    const SliceType& slice,
    typename std::enable_if<
    2==SliceType::kokkos_view::traits::dimension::rank,int*>::type = 0 )
{
    // Reorder in a contiguous blocked format.
    Kokkos::View<typename SliceType::value_type*,
                 typename SliceType::device_type> view( "field", slice.size() );
    Kokkos::parallel_for(
        "SiloParticleWriter::writeFieldRank0",
        Kokkos::RangePolicy<typename SliceType::execution_space>(0,slice.size()),
        KOKKOS_LAMBDA( const int i ){
            view(i) = slice(i);
        });

    // Mirror the field to the host.
    auto host_view = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), view );

    // Buffer the field for libbat so we don't lose the linearized data
    // TODO: Direct AoSoA API
    const size_t n_bytes = host_view.extent(0) * BATTraits<typename SliceType::value_type>::stride(1);
    uint8_t *buf = new uint8_t[n_bytes];
    std::memcpy(buf, host_view.data(), n_bytes);
    data.data.push_back(std::shared_ptr<uint8_t>(buf, std::default_delete<uint8_t[]>()));

    // Add the field.
    bat_io_set_attribute(bat_file,
                         slice.label().c_str(),
                         buf,
                         host_view.extent(0),
                         BATTraits<typename SliceType::value_type>::type(1));
}

// Rank-1 field
template<class SliceType>
void writeFieldsImpl(
    BATParticleState bat_file,
    BATData &data,
    const std::string& mesh_name,
    const SliceType& slice,
    typename std::enable_if<
    3==SliceType::kokkos_view::traits::dimension::rank,int*>::type = 0 )
{
    // Reorder in a contiguous blocked format.
    Kokkos::View<typename SliceType::value_type**,
                 Kokkos::LayoutLeft,
                 typename SliceType::device_type>
        view( "field", slice.size(), slice.extent(2) );
    Kokkos::parallel_for(
        "SiloParticleWriter::writeFieldRank1",
        Kokkos::RangePolicy<typename SliceType::execution_space>(0,slice.size()),
        KOKKOS_LAMBDA( const int i ){
            for ( std::size_t d0 = 0; d0 < slice.extent(2); ++d0 )
                view(i,d0) = slice(i,d0);
        });

    // Mirror the field to the host.
    auto host_view = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), view );

    // Get the data pointers.
    std::vector<typename SliceType::value_type*> ptrs( host_view.extent(1) );
    for ( std::size_t d0 = 0; d0 < host_view.extent(1); ++d0 ) {
        ptrs[d0] = &host_view(0,d0);

        const size_t n_bytes = host_view.extent(0) * BATTraits<typename SliceType::value_type>::stride(1);
        uint8_t *buf = new uint8_t[n_bytes];
        std::memcpy(buf, &host_view(0, d0), n_bytes);
        data.data.push_back(std::shared_ptr<uint8_t>(buf, std::default_delete<uint8_t[]>()));

        // TODO: AoSoA API
        const std::string name = slice.label() + std::to_string(d0);
        bat_io_set_attribute(bat_file,
                             name.c_str(),
                             buf,
                             host_view.extent(0),
                             BATTraits<typename SliceType::value_type>::type(1));
    }
}

#if 0
// Not used in the example
// Rank-2 field
template<class SliceType>
void writeFieldsImpl(
    DBfile* silo_file,
    const std::string& mesh_name,
    const SliceType& slice,
    typename std::enable_if<
    4==SliceType::kokkos_view::traits::dimension::rank,int*>::type = 0 )
{
    // Reorder in a contiguous blocked format.
    Kokkos::View<typename SliceType::value_type***,
                 Kokkos::LayoutLeft,
                 typename SliceType::device_type>
        view( "field", slice.size(), slice.extent(2), slice.extent(3) );
    Kokkos::parallel_for(
        "SiloParticleWriter::writeFieldRank2",
        Kokkos::RangePolicy<typename SliceType::execution_space>(0,slice.size()),
        KOKKOS_LAMBDA( const int i ){
            for ( std::size_t d0 = 0; d0 < slice.extent(2); ++d0 )
                for ( std::size_t d1 = 0; d1 < slice.extent(3); ++d1 )
                    view(i,d0,d1) = slice(i,d0,d1);
        });

    // Mirror the field to the host.
    auto host_view = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), view );

    // Get the data pointers.
    std::vector<typename SliceType::value_type*> ptrs;
    ptrs.reserve( host_view.extent(1) * host_view.extent(2) );
    for ( unsigned d0 = 0; d0 < host_view.extent(1); ++d0 )
        for ( unsigned d1 = 0; d1 < host_view.extent(2); ++d1 )
            ptrs.push_back( &host_view(0,d0,d1) );

    // Write the field.
    /*
    DBPutPointvar( silo_file,
                   slice.label().c_str(),
                   mesh_name.c_str(),
                   host_view.extent(1) * host_view.extent(2), ptrs.data(),
                   host_view.extent(0),
                   SiloTraits<typename SliceType::value_type>::type(),
                   nullptr );
                   */
}
#endif

template<class SliceType>
void writeFields(BATParticleState bat_file,
                 BATData &data,
                 const std::string& mesh_name,
                 const SliceType& slice )
{
    writeFieldsImpl( bat_file, data, mesh_name, slice );
}

template<class SliceType, class ... FieldSliceTypes>
void writeFields(BATParticleState bat_file,
                 BATData &data,
                 const std::string& mesh_name,
                 const SliceType& slice,
                 FieldSliceTypes&&... fields )
{
    writeFieldsImpl( bat_file, data, mesh_name, slice );
    writeFields( bat_file, data, mesh_name, fields... );
}

//---------------------------------------------------------------------------//
// Write a time step.
template<class LocalGridType, class CoordSliceType, class ... FieldSliceTypes>
void writeTimeStep( const LocalGridType& local_grid,
                    const int time_step_index,
                    const double time,
                    const CoordSliceType& coords,
                    FieldSliceTypes&&... fields )
{
    // Reorder the coordinates in a blocked format.
    Kokkos::View<typename CoordSliceType::value_type**,
                 Kokkos::LayoutLeft,
                 typename CoordSliceType::device_type>
        view( "coords", coords.size(), coords.extent(2) );
    Kokkos::parallel_for(
        "BATParticleWriter::writeCoords",
        Kokkos::RangePolicy<typename CoordSliceType::execution_space>(
            0,coords.size()),
        KOKKOS_LAMBDA( const int i ){
            for ( std::size_t d0 = 0; d0 < coords.extent(2); ++d0 )
                view(i,d0) = coords(i,d0);
        });

    // Mirror the coordinates to the host.
    auto host_coords = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), view );

    // Add the point mesh.
    std::string mesh_name = "particles";
    float* ptrs[3] =
        {&host_coords(0,0), &host_coords(0,1), &host_coords(0,2)};

    // Set bounds of our local grid
    auto owned_cells =
        local_grid->indexSpace( Cajita::Own(), Cajita::Cell(), Cajita::Local() );
    int rank_lo[3] = {owned_cells.min(Dim::I), owned_cells.min(Dim::J), owned_cells.min(Dim::K)};
    int rank_hi[3] = {owned_cells.max(Dim::I), owned_cells.max(Dim::J), owned_cells.max(Dim::K)};

    using device_type = Kokkos::HostSpace;
    auto local_mesh = Cajita::createLocalMesh<device_type>( *local_grid );

    int rank = 0;
    int comm_size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
#if 0
    std::cout << "rank " << rank << " has cells: {" << rank_lo[0] << ", " << rank_lo[1]
        << ", " << rank_lo[2] << "} to {" << rank_hi[0] << ", " << rank_hi[1] << ", "
        << rank_hi[2] << "}\n";
#endif

    double low_coords[3];
    local_mesh.coordinates( Cajita::Node(), rank_lo, low_coords );
    double hi_coords[3];
    local_mesh.coordinates( Cajita::Node(), rank_hi, hi_coords );
#if 0
    std::cout << "Spatial bounds: {" << low_coords[0] << ", " << low_coords[1]
        << ", " << low_coords[2] << "}, {"
        << hi_coords[0] << ", " << hi_coords[1] << ", " << hi_coords[2] << "}\n";
#endif
    const float rank_bounds[6] = {
        low_coords[0], low_coords[1], low_coords[2],
        hi_coords[0], hi_coords[1], hi_coords[2]
    };

    // TODO: AoSoA support for writes
    // Here we need to transform the data into an SoA layout for now
    // this could probably be done in the kokkos lambda, but for a hack test this is ok
    std::shared_ptr<float> positions(new float[host_coords.extent(0) * 3], std::default_delete<float[]>());
    float *positions_buf = positions.get();
    for (size_t i = 0; i < host_coords.extent(0); ++i) {
        positions_buf[i * 3] = host_coords(i, 0);
        positions_buf[i * 3 + 1] = host_coords(i, 1);
        positions_buf[i * 3 + 2] = host_coords(i, 2);
    }

    auto bat_file = bat_io_allocate();

    bat_io_set_local_bounds(bat_file, rank_bounds);
    bat_io_set_positions(bat_file,
                         positions_buf,
                         host_coords.extent(0),
                         BTD_VEC3_FLOAT);

    // Add variables.
    BATData data;
    writeFields( bat_file, data, mesh_name, fields... );

    using namespace std::chrono;

    const uint64_t local_particles = host_coords.extent(0);
    uint64_t global_particles = 0;
    MPI_Allreduce(&local_particles, &global_particles, 1,
            MPI_UNSIGNED_LONG_LONG,
            MPI_SUM,
            MPI_COMM_WORLD);

    // Each particle is ~36 bytes
    const size_t particle_size = 36;
    const size_t global_bytes = global_particles * particle_size;
    const size_t fpp_size = global_bytes / comm_size + global_bytes % comm_size;

    std::string file_name;
    if (getenv("OUTPUT_FILE")) {
        file_name = getenv("OUTPUT_FILE");
    } else {
        file_name = "particles";
    }

    if (getenv("JOB_ID") == nullptr) {
        throw std::runtime_error("JOB_ID must be set in the environment");
    }
    const std::string job_id = getenv("JOB_ID");

    const int do_fixed_agg = getenv("FIXED_AGGREGATION") != nullptr ? 1 : 0;
    const int do_dump_raw = getenv("DUMP_RAW") != nullptr ? 1 : 0;

    // Run a set of write scaling experiments so we can do the tests in a single job
    const int max_agg_steps = std::min(static_cast<int>(std::log2(comm_size)), 5);
    for (int agg_step = 0; agg_step <= max_agg_steps; ++agg_step) {
        for (int fixed_agg = 0; fixed_agg <= do_fixed_agg; ++fixed_agg) {
            for (int dump_raw = 0; dump_raw <= do_dump_raw; ++dump_raw) {
                const size_t agg_size = (1 << agg_step) * fpp_size;

                std::string log_file = "dambreak-" + std::to_string(agg_step)
                    + "-" + job_id;
                if (rank == 0) {
                    std::cout << "Aggregation size: " << agg_size
                        << " (avg. fpp size: " << fpp_size << ")\n";

                    if (dump_raw) {
                        std::cout << "Dump raw\n";
                        log_file += "-raw";
                    }
                    if (fixed_agg) {
                        std::cout << "Fixed aggregation\n";
                        log_file += "-fixed";
                    }
                    log_file += ".out";
                    std::cout << "Logged to " << log_file << "\n";
                }

                bat_io_set_bytes_per_subfile(bat_file, agg_size);
                bat_io_set_build_local_trees(bat_file, !dump_raw);
                bat_io_set_fixed_aggregation(bat_file, fixed_agg);

                std::string test_file_name = file_name + "-" + std::to_string(agg_step)
                    + "-t" + std::to_string(time_step_index);
                if (dump_raw) {
                    test_file_name += "-raw";
                }
                if (fixed_agg) {
                    test_file_name += "-fixed";
                }

                auto start = high_resolution_clock::now();
                const uint64_t bytes_written = bat_io_write(bat_file, test_file_name.c_str());
                auto end = high_resolution_clock::now();
                const char *perf_stats = bat_io_get_performance_statistics(bat_file);
                if (rank == 0) {
                    std::ofstream log_out(log_file.c_str(), std::ios::app | std::ios::out);

                    const size_t write_time = duration_cast<milliseconds>(end - start).count();
                    const float bandwidth = (bytes_written * 1e-6f) / (write_time * 1e-3f);
                    log_out << "Timestep: " << time_step_index << "\n"
                        << "Target size: " << agg_size << "b\n"
                        << "Total write time: " << write_time << "ms\n"
                        << "Total bytes written: " << bytes_written << "b\n"
                        << "Write bandwidth: " << bandwidth << "MB/s\n"
                        << "Perf Stats: " << perf_stats << "\n"
                        << "=======\n"
                        << std::flush;
                }
            }
        }
    }

    bat_io_free(bat_file);
}

//---------------------------------------------------------------------------//

} // end namespace BATParticleWriter
} // end namespace ExaMPM

#endif // EXAMPM_BATPARTICLEWRITER_HPP

