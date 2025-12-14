#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#include "hdf5.h"
#ifdef CONTAINS_MPI
    #include "mpi.h"
#endif
#include "selected_kernel.h"

hid_t open_hdf5_file(char *file) {
    return H5Fopen(file, H5F_ACC_RDONLY, H5P_DEFAULT);
}
hid_t open_hdf5_dataset(hid_t file_id, char *dataset) {
    return H5Dopen2(file_id, dataset, H5P_DEFAULT);
}
f_type *read_float_dataset(hid_t dataset_id) {
    hid_t dataspace;
    hsize_t dims_out[10];
    int rank, total_size;

    dataspace = H5Dget_space(dataset_id);
    rank = H5Sget_simple_extent_ndims(dataspace);
    H5Sget_simple_extent_dims(dataspace, dims_out, NULL);

    total_size = 1;
    for (size_t i = 0; i < rank; i++) {
        total_size *= dims_out[i];
    }

    // Detecta o tipo de dados do dataset
    hid_t datatype = H5Dget_type(dataset_id);
    hid_t native_type = H5Tget_native_type(datatype, H5T_DIR_ASCEND);
    H5T_class_t type_class = H5Tget_class(native_type);
    
    f_type *dset_data = malloc(sizeof(f_type) * total_size);
    hid_t mem_type;
    
    #if defined(DOUBLE)
        mem_type = H5T_NATIVE_DOUBLE;
    #else
        mem_type = H5T_NATIVE_FLOAT;
    #endif
    
    H5Dread(dataset_id, mem_type, H5S_ALL, H5S_ALL, H5P_DEFAULT, dset_data);
    
    H5Tclose(native_type);
    H5Tclose(datatype);
    H5Sclose(dataspace);
    
    return dset_data;
}
f_type read_float_attribute(hid_t dataset_id, char *attribute_name) {
    const char *attribute_value;
    hid_t attribute_id = H5Aopen(dataset_id, attribute_name, H5P_DEFAULT);
    hid_t attribute_type = H5Aget_type(attribute_id);
    H5Aread(attribute_id, attribute_type, &attribute_value);
    #if defined(DOUBLE)
        return atof(attribute_value);
    #else
        return (float)atof(attribute_value);
    #endif
}
void close_hdf5_dataset(hid_t dataset_id) { H5Dclose(dataset_id); }
void close_hdf5_file(hid_t file_id) { H5Fclose(file_id); }

int write_hdf5_result(int n1, int n2, int n3, double execution_time, f_type* next_u) {
    hid_t h5t_type = H5T_NATIVE_FLOAT;
    #if defined(DOUBLE)
        h5t_type = H5T_NATIVE_DOUBLE;
    #endif

    hid_t file_id = H5Fcreate("c-frontend/data/results.h5", H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (file_id < 0) {
        printf("Error creating file.\n");
        return 1;
    }

    hsize_t vector_dims[3] = {n1,n2,n3};
    hid_t vector_dataspace_id = H5Screate_simple(3, vector_dims, NULL);
    if (vector_dataspace_id < 0) {
        printf("Error creating vector dataspace.\n");
        H5Fclose(file_id);
        return 1;
    }

    hid_t vector_dataset_id = H5Dcreate(file_id, "vector", h5t_type, vector_dataspace_id,
                                        H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (vector_dataset_id < 0) {
        printf("Error creating vector dataset.\n");
        H5Sclose(vector_dataspace_id);
        H5Fclose(file_id);
        return 1;
    }

    if (H5Dwrite(vector_dataset_id, h5t_type, H5S_ALL, H5S_ALL, H5P_DEFAULT, next_u) < 0) {
        printf("Error writing vector data.\n");
        H5Dclose(vector_dataset_id);
        H5Sclose(vector_dataspace_id);
        H5Fclose(file_id);
        return 1;
    }

    hsize_t time_dims[1] = {1};
    hid_t time_dataspace_id = H5Screate_simple(1, time_dims, NULL);
    if (time_dataspace_id < 0) {
        printf("Error creating time dataspace.\n");
        H5Dclose(vector_dataset_id);
        H5Sclose(vector_dataspace_id);
        H5Fclose(file_id);
        return 1;
    }

    hid_t time_dataset_id = H5Dcreate(file_id, "execution_time", H5T_NATIVE_DOUBLE, time_dataspace_id,
                                      H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (time_dataset_id < 0) {
        printf("Error creating time dataset.\n");
        H5Dclose(vector_dataset_id);
        H5Sclose(vector_dataspace_id);
        H5Sclose(time_dataspace_id);
        H5Fclose(file_id);
        return 1;
    }

    if (H5Dwrite(time_dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &execution_time) < 0) {
        printf("Error writing time data.\n");
        H5Dclose(vector_dataset_id);
        H5Sclose(vector_dataspace_id);
        H5Dclose(time_dataset_id);
        H5Sclose(time_dataspace_id);
        H5Fclose(file_id);
        return 1;
    }

    H5Dclose(vector_dataset_id);
    H5Sclose(vector_dataspace_id);
    H5Dclose(time_dataset_id);
    H5Sclose(time_dataspace_id);
    H5Fclose(file_id);
    return 0;
}

void print_wavefield_analysis(f_type *data, int n1, int n2, int n3, const char *label) {
    int n_total = n1 * n2 * n3;
    f_type min_val = data[0], max_val = data[0];
    double sum = 0.0;
    int non_zero_count = 0;
    f_type threshold = 0.01;

    for (int i = 0; i < n_total; i++) {
        if (data[i] < min_val) min_val = data[i];
        if (data[i] > max_val) max_val = data[i];
        sum += data[i];
        if (fabs(data[i]) > threshold) {
            non_zero_count++;
        }
    }

    f_type percentage = (non_zero_count * 100.0) / n_total;
    
    printf("=== %s ===\n", label);
    printf("Min: %.6f | Max: %.6f | Sum: %.2f | Mean: %.6f\n", 
           (double)min_val, (double)max_val, sum, sum/n_total);
    printf("Active elements (|u| > %.2f): %d / %d (%.2f%% of grid)\n", 
           (double)threshold, non_zero_count, n_total, (double)percentage);
    printf("\n");
}

void print_central_slice(f_type *data, int n1, int n2, int n3, const char *label) {
    int center_z = n1 / 2;
    int center_y = n2 / 2;
    
    printf("=== %s - CENTRAL CROSS-SECTION (y=%d plane) ===\n", label, center_y);
    printf("Line from center: [z=%d, y=%d, x=0:31]\n", center_z, center_y);
    
    for (int x = 0; x < 32 && x < n3; x++) {
        int idx = center_z * n2 * n3 + center_y * n3 + x;
        printf("%7.3f ", (double)data[idx]);
        if ((x + 1) % 8 == 0) printf("\n");
    }
    printf("\n");
}

int main() {
    #ifdef CONTAINS_MPI
        MPI_Init(NULL, NULL);
    #endif

    int rank = 0;

    #ifdef CONTAINS_MPI
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    #endif

    hid_t file_id = open_hdf5_file("c-frontend/data/miniwave_data.h5");
    hid_t vel_model_id = open_hdf5_dataset(file_id, "vel_model");
    hid_t next_u_id = open_hdf5_dataset(file_id, "next_u");
    hid_t prev_u_id = open_hdf5_dataset(file_id, "prev_u");
    hid_t coefficient_id = open_hdf5_dataset(file_id, "coefficient");
    hid_t scalar_data_id = open_hdf5_dataset(file_id, "scalar_data");
    f_type *vel_model = read_float_dataset(vel_model_id);
    f_type *next_u = read_float_dataset(next_u_id);
    f_type *prev_u = read_float_dataset(prev_u_id);
    f_type *coefficient = read_float_dataset(coefficient_id);
    f_type block_size_1 = read_float_attribute(scalar_data_id, "block_size_1");
    f_type block_size_2 = read_float_attribute(scalar_data_id, "block_size_2");
    f_type block_size_3 = read_float_attribute(scalar_data_id, "block_size_3");
    f_type d1 = read_float_attribute(scalar_data_id, "d1");
    f_type d2 = read_float_attribute(scalar_data_id, "d2");
    f_type d3 = read_float_attribute(scalar_data_id, "d3");
    f_type dt = read_float_attribute(scalar_data_id, "dt");
    f_type iterations = read_float_attribute(scalar_data_id, "iterations");
    f_type n1 = read_float_attribute(scalar_data_id, "n1");
    f_type n2 = read_float_attribute(scalar_data_id, "n2");
    f_type n3 = read_float_attribute(scalar_data_id, "n3");
    f_type stencil_radius = read_float_attribute(scalar_data_id, "stencil_radius");

    if (rank == 0) {
        printf("\n");
        printf("╔════════════════════════════════════════════════════════════╗\n");
        printf("║           MINIWAVE - 3D ACOUSTIC WAVE SIMULATOR            ║\n");
        printf("╚════════════════════════════════════════════════════════════╝\n");
        printf("\n");
        printf(" GRID CONFIGURATION:\n");
        printf("   Dimensions: %.0f x %.0f x %.0f = %.0f points\n", 
               n1, n2, n3, n1*n2*n3);
        printf("   Spacing: %.1f x %.1f x %.1f meters\n", d1, d2, d3);
        printf("   Physical size: %.0f x %.0f x %.0f meters\n", 
               n1*d1, n2*d2, n3*d3);
        printf("\n");
        printf("CONFIGURATION:\n");
        printf("   Timesteps: %.0f\n", iterations);
        printf("   Time step (dt): %.6f seconds\n", dt);
        printf("   Total simulated time: %.6f seconds\n", iterations * dt);
        printf("\n");
        printf(" NUMERICAL METHOD:\n");
        printf("   Stencil radius: %.0f (space order: %.0f)\n", 
               stencil_radius, stencil_radius*2);
        printf("   FD coefficients: [%.3f, %.3f]\n", 
               coefficient[0], coefficient[1]);
        printf("\n");
        printf(" PU CONFIGURATION:\n");
        printf("   Block size: %.0f x %.0f x %.0f threads\n", 
               block_size_1, block_size_2, block_size_3);
        printf("\n");

        print_wavefield_analysis(prev_u, n1, n2, n3, "INITIAL WAVEFIELD");
        print_central_slice(prev_u, n1, n2, n3, "INITIAL STATE");
    }

    printf("╔════════════════════════════════════════════════════════════╗\n");
    printf("║                    RUNNING SIMULATION...                   ║\n");
    printf("╚════════════════════════════════════════════════════════════╝\n");
    
    clock_t start_time = clock();

    forward(prev_u, next_u, vel_model, coefficient, d1, d2, d3, dt, n1, n2, n3, iterations, stencil_radius, block_size_1, block_size_2, block_size_3);

    clock_t end_time = clock();
    double execution_time = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;

    if (rank == 0) {
        printf("\n");
        printf("╔════════════════════════════════════════════════════════════╗\n");
        printf("║                  SIMULATION COMPLETED! ✓                   ║\n");
        printf("╚════════════════════════════════════════════════════════════╝\n");
        printf("\n");

        print_wavefield_analysis(next_u, n1, n2, n3, "FINAL WAVEFIELD");
        print_central_slice(next_u, n1, n2, n3, "FINAL STATE");

        int center_idx = ((int)n1/2) * (int)n2 * (int)n3 + ((int)n2/2) * (int)n3 + ((int)n3/2);
        printf(" CENTER POINT [%d,%d,%d]:\n", (int)n1/2, (int)n2/2, (int)n3/2);
        printf("   Before: %.6f → After: %.6f (change: %+.6f)\n", 
               (double)prev_u[center_idx], (double)next_u[center_idx], 
               (double)(next_u[center_idx] - prev_u[center_idx]));
        printf("\n");

        write_hdf5_result((int)n1, (int)n2, (int)n3, execution_time, next_u);
        
        printf(" PERFORMANCE:\n");
        printf("   Execution time: %.6f seconds\n", execution_time);
        printf("   Points computed: %.0f\n", n1 * n2 * n3 * iterations);
        printf("   Throughput: %.2f million points/second\n", 
               (n1 * n2 * n3 * iterations / execution_time) / 1e6);
        printf("\n");
        printf(" Results saved to: c-frontend/data/results.h5\n");
        printf("\n");
    }

    close_hdf5_dataset(vel_model_id);
    close_hdf5_dataset(next_u_id);
    close_hdf5_dataset(prev_u_id);
    close_hdf5_dataset(coefficient_id);
    close_hdf5_file(file_id);

    #ifdef CONTAINS_MPI
        MPI_Finalize();
    #endif
}
