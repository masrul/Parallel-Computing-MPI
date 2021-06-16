// Standard Includes for MPI, C and OS calls
#include <algorithm>
#include <limits.h>
#include <mpi.h>
#include <signal.h>
#include <stdlib.h>
#include <unistd.h>
// C++ standard I/O calls
#include <iostream>
using std::cerr;
using std::cout;
using std::endl;

#include <cmath>
using std::pow;

int sleep_time = 540; // 9 minutes

//**************************************************************************/
// UTILITY ROUTINES

// This routine handles program traps.  We put this in here so that the
// mpi program will be more robust to crashes and won't lock up the queue

void program_trap(int sig) /*,code,scp,addr)*/ {
    const char* sigtype = "(undefined)";

    switch (sig) {
    case SIGBUS:
        sigtype = "a Bus Error";
        break;
    case SIGSEGV:
        sigtype = "a Segmentation Violation";
        break;
    case SIGILL:
        sigtype = "an Illegal Instruction Call";
        break;
    case SIGSYS:
        sigtype = "an Illegal System Call";
        break;
    case SIGFPE:
        sigtype = "a Floating Point Exception";
        break;
    case SIGALRM:
        sigtype = "a Alarm Signal!";
        break;
    }
    fprintf(stderr, "ERROR: Program terminated due to %s\n", sigtype);
    abort();
}

// This routine redirects the OS signal traps to the above routine
// Also sets a 9 minute alarm to kill the program in case it runs
// away.  This really should prevent job queue problems for errant
// programs.
void chopsigs_() {
    signal(SIGBUS, program_trap);
    signal(SIGSEGV, program_trap);
    signal(SIGILL, program_trap);
    signal(SIGSYS, program_trap);
    signal(SIGFPE, program_trap);
    signal(SIGALRM, program_trap);
    // Send an alarm signal after 9 minutes elapses
    alarm(sleep_time);
}

// A utility routine for measuring time
double get_timer() {
    static double to = 0;
    double tn, t;
    tn = MPI_Wtime();
    t = tn - to;
    to = tn;
    return t;
}

// function that implements 2^i
inline int pow2(int p) { return 1 << p; }

// function that implements log_2(i)
inline int log2(int v) {
    int d = 0;
    for (v = v >> 1; v != 0; v = v >> 1)
        d++;
    return d;
}

// Binary search of lower bound
int lower_bound(double a[], int n, double x) {
    int low = 0;
    int high = n; // Not n - 1
    while (low < high) {
        int mid = (low + high) / 2;
        if (x <= a[mid]) {
            high = mid;
        } else {
            low = mid + 1;
        }
    }
    return low;
}

//************************************************************************/
// Communication Variables

// We will use these routine to store the size and rank in MPI_COMM_WORLD
int numprocs, myid;


//***********************************************************************/
// Example implementation of bitonic sort

// Below are the compare split functions for use in the bitonic sort.
// They take in three buffers, a local buffer, and two temporary buffers
// They use pointer swapping to avoid copying.
void compare_split_max(double** local_buf, double** recv_buf,
                       double** result_buf, int loc_size, int max_buf_size,
                       int dest_addr) {
    MPI_Status stat;

    MPI_Sendrecv(*local_buf, loc_size, MPI_DOUBLE, dest_addr, 0, *recv_buf,
                 max_buf_size, MPI_DOUBLE, dest_addr, 0, MPI_COMM_WORLD, &stat);

    int recv_size;
    MPI_Get_count(&stat, MPI_DOUBLE, &recv_size);

    int ld = loc_size - 1, rd = recv_size - 1;
    for (int i = loc_size - 1; i >= 0; --i) {
        if (rd < 0)
            (*result_buf)[i] = (*local_buf)[ld--];
        else if (ld < 0)
            (*result_buf)[i] = (*recv_buf)[rd--];
        else if ((*local_buf)[ld] > (*recv_buf)[rd])
            (*result_buf)[i] = (*local_buf)[ld--];
        else
            (*result_buf)[i] = (*recv_buf)[rd--];
    }

    std::swap(*result_buf, *local_buf);
}

void compare_split_min(double** local_buf, double** recv_buf,
                       double** result_buf, int loc_size, int max_buf_size,
                       int dest_addr) {
    MPI_Status stat;
    MPI_Sendrecv(*local_buf, loc_size, MPI_DOUBLE, dest_addr, 0, *recv_buf,
                 max_buf_size, MPI_DOUBLE, dest_addr, 0, MPI_COMM_WORLD, &stat);

    int recv_size;
    MPI_Get_count(&stat, MPI_DOUBLE, &recv_size);

    int ld = 0, rd = 0;
    for (int i = 0; i < loc_size; ++i) {
        if (rd == recv_size)
            (*result_buf)[i] = (*local_buf)[ld++];
        else if (ld == loc_size)
            (*result_buf)[i] = (*recv_buf)[rd++];
        else if ((*local_buf)[ld] < (*recv_buf)[rd])
            (*result_buf)[i] = (*local_buf)[ld++];
        else
            (*result_buf)[i] = (*recv_buf)[rd++];
    }
    std::swap(*result_buf, *local_buf);
}

// Parallel bitonic sort algorithm as described in the text.
double* parallel_bitonic_sort(double* buffer, int& loc_buf_size, int max_size) {
    if (numprocs & (numprocs - 1)) {
        cerr << "bitonic sort requires 2^d processors" << endl;
        MPI_Abort(MPI_COMM_WORLD, -1);
        abort();
    }

    // First sort the local sequence
    std::sort(buffer, buffer + loc_buf_size);

    // Allocate some buffers ;
    double* recv_buffer = new double[max_size];
    double* result_buffer = new double[max_size];

    // Perform bitonic sort on a d dimensional hypercube
    int d = log2(numprocs);

    for (int i = 0; i < d; ++i) {
        for (int j = i; j >= 0; --j) {
            int ibit = (myid & pow2(i + 1)) != 0;
            int jbit = (myid & pow2(j)) != 0;
            int dest_addr = myid ^ pow2(j);
            if (ibit != jbit)
                compare_split_max(&buffer, &recv_buffer, &result_buffer,
                                  loc_buf_size, max_size, dest_addr);
            else
                compare_split_min(&buffer, &recv_buffer, &result_buffer,
                                  loc_buf_size, max_size, dest_addr);
        }
    }

    delete[] recv_buffer;
    delete[] result_buffer;
    return buffer;
}

double* parallel_sample_native_sort(double* buffer, int& loc_buf_size,
                                    MPI_Comm comm) {

    int i, j;

    /* Get communicator info */
    MPI_Comm_size(comm, &numprocs);
    MPI_Comm_rank(comm, &myid);

    // nlocal=loc_buf_size;
    int splitter_size = numprocs - 1;
    double* splitters = new double[numprocs];
    double* allpicks = new double[numprocs * (numprocs - 1)];

    /* sort local array */
    std::sort(buffer, buffer + loc_buf_size);

    /* Select local npes-1 equally spaced elements */
    for (i = 1; i < numprocs; ++i)
        splitters[i - 1] = buffer[i * loc_buf_size / numprocs];

    /* Gather the sample in the processors */
    MPI_Allgather(splitters, numprocs - 1, MPI_INT, allpicks, numprocs - 1,
                  MPI_INT, comm);

    /* sort these samples */
    std::sort(allpicks, allpicks + numprocs * (numprocs - 1));

    /* Select splitters */
    for (i = 1; i < numprocs; ++i)
        splitters[i - 1] = allpicks[i + numprocs];
    splitters[numprocs - 1] = INT_MAX;

    /* comput the number of elements that belong to each bucket */
    int* scounts = new int[numprocs];
    for (i = 0; i < numprocs; ++i)
        scounts[i] = 0;

    for (j = i = 0; i < loc_buf_size; ++i) {
        if ((j == numprocs - 1) || (buffer[i] < splitters[j]))
            scounts[j]++;
        else {
            // skip empty slots
            while ((j != numprocs - 1) && !(buffer[i] < splitters[j]))
                j++;
            scounts[j]++;
        }
    }
    /*Determine the starting location of each bucket's elements in the
      element array */
    int* sdispls = new int[numprocs];
    sdispls[0] = 0;
    for (i = 1; i < numprocs; ++i)
        sdispls[i] = sdispls[i - 1] + scounts[i - 1];

    /*Perform an all-to-all to inform the corresponding process of
     the number of elements. They are going to receive . This information is
     stored in rcounts array */

    int* rcounts = new int[numprocs];
    MPI_Alltoall(scounts, 1, MPI_INT, rcounts, 1, MPI_INT, comm);

    /*Based on rcounts determine where in the local array the data from each
     * process will be stored. This array will store the received elements as
     * well as the fina sorted sequence */

    int* rdispls = new int[numprocs];
    rdispls[0] = 0;
    for (i = 1; i < numprocs; ++i)
        rdispls[i] = rdispls[i - 1] + rcounts[i - 1];

    loc_buf_size = rdispls[i - 1] + rcounts[i - 1];
    double* sorted_buffer = new double[loc_buf_size];

    MPI_Alltoallv(buffer, scounts, sdispls, MPI_INT, sorted_buffer, rcounts,
                  rdispls, MPI_INT, comm);

    /*Perform the final local sort*/
    std::sort(sorted_buffer, sorted_buffer + loc_buf_size);

    delete[] splitters;
    delete[] allpicks;
    delete[] scounts;
    delete[] sdispls;
    delete[] rcounts;
    delete[] rdispls;

    return sorted_buffer;
}

double* parallel_sample_bitonic_sort(double* buffer, int& loc_buf_size,
                                     MPI_Comm comm) {
    int i, j;

    /* Get communicator info */
    MPI_Comm_size(comm, &numprocs);
    MPI_Comm_rank(comm, &myid);

    int splitter_size = numprocs;
    double* splitters = new double[numprocs];
    double* allpicks = new double[numprocs];

    /* sort local array */
    std::sort(buffer, buffer + loc_buf_size);

    /* Select local numprocs-1 equally spaced elements */
    for (i = 1; i < numprocs; i++)
        splitters[i - 1] = buffer[i * loc_buf_size / numprocs];

    splitters = parallel_bitonic_sort(splitters, splitter_size, splitter_size);

    // MPI_Allgather
    MPI_Allgather(&splitters[splitter_size / 2], 1, MPI_DOUBLE, allpicks, 1,
                  MPI_DOUBLE, comm);
    allpicks[numprocs - 1] = INT_MAX;

    /* comput the number of elements that belong to each bucket */
    int* scounts = new int[numprocs];
    for (i = 0; i < numprocs; i++)
        scounts[i] = 0;

    for (j = i = 0; i < loc_buf_size; ++i) {
        if ((j == numprocs - 1) || (buffer[i] < allpicks[j]))
            scounts[j]++;
        else {
            // skip empty slots
            while ((j != numprocs - 1) && !(buffer[i] < allpicks[j]))
                j++;
            scounts[j]++;
        }
    }

    /*Determine the starting location of each bucket's elements in the
      element array */
    int* sdispls = new int[numprocs];
    sdispls[0] = 0;
    for (i = 1; i < numprocs; i++)
        sdispls[i] = sdispls[i - 1] + scounts[i - 1];

    /*Perform an all-to-all to inform the corresponding process of
     the number of elements. They are going to receive . This information is
     stored in rcounts array */

    int* rcounts = new int[numprocs];
    MPI_Alltoall(scounts, 1, MPI_INT, rcounts, 1, MPI_INT, comm);

    /*Based on rcounts determine where in the local array the data from each
     * process will be stored. This array will store the received elements as
     * well as the fina sorted sequence */

    int* rdispls = new int[numprocs];
    rdispls[0] = 0;
    for (i = 1; i < numprocs; i++)
        rdispls[i] = rdispls[i - 1] + rcounts[i - 1];

    loc_buf_size = rdispls[i - 1] + rcounts[i - 1];
    double* sorted_buffer = new double[loc_buf_size];

    MPI_Alltoallv(buffer, scounts, sdispls, MPI_DOUBLE, sorted_buffer, rcounts,
                  rdispls, MPI_DOUBLE, comm);

    /*Perform the final local sort*/
    std::sort(sorted_buffer, sorted_buffer + loc_buf_size);

    delete[] splitters;
    delete[] allpicks;
    delete[] scounts;
    delete[] sdispls;
    delete[] rcounts;
    delete[] rdispls;

    return sorted_buffer;
}

double* parallel_quick_sort(double* buffer, int& loc_buf_size, MPI_Comm comm) {
    if (numprocs & (numprocs - 1)) {
        cerr << "Quick sort requires 2^d processors" << endl;
        MPI_Abort(MPI_COMM_WORLD, -1);
        abort();
    }

    // Allocate some buffers ;
    int max_size = (loc_buf_size + 1) * numprocs;
    double* recv_buffer = new double[max_size];
    double* result_buffer = new double[max_size];
    double* median_buffer = new double[numprocs];
    double my_median;

    // Put buffer into result buffer
    for (int i = 0; i < loc_buf_size; ++i) {
        result_buffer[i] = buffer[i];
    }
    int result_size = loc_buf_size;

    // Perform quick sort on a d dimensional hypercube
    int d = log2(numprocs);
    int new_comm_size, my_new_id;
    for (int i = 0; i < d; ++i) {

        // Determine number of sub-groups and colors
        int num_sub_grps = pow2(i);
        int color = myid / pow2(d - i);

        // Sort the local list followed by determining median
        std::sort(result_buffer, result_buffer + result_size);
        my_median = result_buffer[result_size / 2];

        // Split the communicator
        MPI_Comm new_comm;
        MPI_Status status;
        MPI_Comm_split(comm, color, myid, &new_comm);
        MPI_Comm_size(new_comm, &new_comm_size);
        MPI_Comm_rank(new_comm, &my_new_id);
        int new_d = log2(new_comm_size);

        int median_size = 1;

        // Gather all median onto all rank
        MPI_Allgather(&my_median, 1, MPI_DOUBLE, median_buffer, 1, MPI_DOUBLE,
                      new_comm);

        // Sort and determine the pivot
        std::sort(median_buffer, median_buffer + new_comm_size);
        double pivot = median_buffer[new_comm_size / 2];

        // Determine pivot position
        int pivot_index = lower_bound(result_buffer, result_size, pivot);

        // Determine whom i am going to  exchange data
        int my_partner = my_new_id ^ (pow2(new_d - 1));

        int actual_recv_count;
        int send_index;
        int send_size;
        int recv_index;
        int recv_size;

        if (my_new_id < my_partner) {
            if (pivot_index <= result_size - 1) {
                send_index = pivot_index;
                send_size = result_size - pivot_index;
                recv_index = pivot_index;
            } else {
                send_index = 0;
                send_size = 0; // Just send nothing
                recv_index = result_size;
            }
            recv_size = (max_size - (result_size - send_size));
            MPI_Send(&result_buffer[send_index], send_size, MPI_DOUBLE,
                     my_partner, 100, new_comm);
            MPI_Recv(&result_buffer[recv_index], recv_size, MPI_DOUBLE,
                     my_partner, 100, new_comm, &status);
            MPI_Get_count(&status, MPI_DOUBLE, &actual_recv_count);
            result_size = result_size + actual_recv_count - send_size;
        } else {
            send_index = 0;
            if (pivot_index > result_size - 1) {
                send_size = result_size;
            } else {
                send_size = pivot_index;
            }

            recv_size = max_size - result_size;
            MPI_Recv(&recv_buffer[0], recv_size, MPI_DOUBLE, my_partner, 100,
                     new_comm, &status);
            MPI_Send(&result_buffer[send_index], send_size, MPI_DOUBLE,
                     my_partner, 100, new_comm);
            MPI_Get_count(&status, MPI_DOUBLE, &actual_recv_count);
            // Merge my result and recv buffer into result buffer
            int j = 0;
            for (int i = pivot_index; i < result_size; ++i) {
                result_buffer[j] = result_buffer[i];
                ++j;
            }
            for (int i = 0; i < actual_recv_count; ++i) {
                result_buffer[j] = recv_buffer[i];
                ++j;
            }
            result_size = result_size + actual_recv_count - send_size;
        }
        MPI_Comm_free(&new_comm);
    }
    loc_buf_size = result_size;
    std::sort(result_buffer, result_buffer + result_size);
    delete[] recv_buffer;
    delete[] median_buffer;
    return result_buffer;
}
//***************************************************************************/
// Program Testing

// Check to see if the sequence has been sorted.  We do this by counting the
// number of times that val[i]>val[i+1].  This will be zero when the sequence
// is properly sorted.
void check_sort(double local_numbers[], int local_size) {
    int check_local = 0;
    for (int i = 0; i < local_size - 1; ++i)
        if (local_numbers[i] > local_numbers[i + 1])
            check_local++;
    if (myid != 0) {
        MPI_Status stat;
        double lowval;
        MPI_Recv(&lowval, 1, MPI_DOUBLE, myid - 1, 0, MPI_COMM_WORLD, &stat);
        if (lowval > local_numbers[0]) {
            check_local++;
        }
    }
    if (myid + 1 != numprocs) { // last processor doesn't have anyone to send to
        MPI_Send(&local_numbers[local_size - 1], 1, MPI_DOUBLE, myid + 1, 0,
                 MPI_COMM_WORLD);
    }

    int check;
    MPI_Reduce(&check_local, &check, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    if (myid == 0) {
        cout << check << " errors in sorting" << endl;
    }
}

//****************************************************************************/
// Main code starts here.

main(int argc, char** argv) {
    double time_passes, max_time;
    /* Initialize MPI */
    MPI_Init(&argc, &argv);

    // Handle signals (Segmentation faults, etc.  Make sure to call MPI cleanup
    // routines when things go wrong.
    chopsigs_();

    /* Get the number of processors and my processor identification */
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    int input_size = 1024; // For debugging we will sort 1024 numbers
    sleep_time = 120;

    if (argc == 2) {
        input_size = atoi(argv[1]);
        sleep_time = 540;
    }

    // Output diagnostic indicating start of operations
    if (0 == myid) {
        cout << "Starting " << numprocs << " processors." << endl;
        cout << "generating input sequence consisting of " << input_size
             << " doubles." << endl;
    }

    // Compute the number of numbers that will be allocated to this processor
    // This estimates the size of the initial buffer sizes.  (Note, this doesn't
    // estimate final buffer sizes!
    int local_input_size = input_size / numprocs;
    int max_local_size = local_input_size + 1;

    // spread the remainder evenly over processors.
    int remainder = input_size % numprocs;
    if (myid < remainder)
        local_input_size += 1;

    // Allocate space for the local numbers
    double* local_numbers = new double[max_local_size];

    // Call MPI_Barrier to before measuring time.  It doesn't assure time sync
    // but its better than nothing.
    MPI_Barrier(MPI_COMM_WORLD);

    /* Every call to get_timer resets the stopwatch.  The next call to get_timer
       will return the amount of time since now */
    get_timer();

    // Now we are about to generate a random sequence of numbers.  To make
    // performance measurements fair, we will generate the numbers in such
    // a way that the sequence will always be the same regardless of the number
    // of processors.  We do this by passing the random number seed between
    // processors.  This approach works but isn't parallel.  Parallel random
    // number generation is an interesting problem, but not the one this project
    // is investigating.

    // First we recieve the seed to begin computing our random numbers
    // locally.

    // The root processor generates the initial seed
    static unsigned short xi[4] = {0, 0, 1, 0};

    // Get the seed to what the previous processor had computed if I am not the
    // root processor.
    if (myid != 0) {
        MPI_Status stat;
        MPI_Recv(xi, 4, MPI_UNSIGNED_SHORT, myid - 1, 0, MPI_COMM_WORLD, &stat);
    }

    // Distribution type, comment out ODD_DIST to get a uniform distribution
    // Undergradutes comment this line, graduates leave it in.
#define ODD_DIST

    for (int i = 0; i < local_input_size; ++i) {
        xi[3] += 1;
        double val = erand48(xi);
#ifdef ODD_DIST
        double p = double(xi[3]) / double(input_size);
        val = pow(val, 1.0 + 3 * p);
        val = val * val;
#endif
        local_numbers[i] = val;
    }

    // Send current seed to next processors
    if (myid + 1 != numprocs) { // last processor doesn't have anyone to send to
        MPI_Send(xi, 4, MPI_UNSIGNED_SHORT, myid + 1, 0, MPI_COMM_WORLD);
    }

    // Synchronize for a time measurement.
    MPI_Barrier(MPI_COMM_WORLD);

    /* Every call to get_timer resets the stopwatch.  The next call to get_timer
       will return the amount of time since now */
    double seq_gen_time = get_timer();

    MPI_Reduce(&seq_gen_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0,
               MPI_COMM_WORLD);

    if (myid == 0) {
        cout << "completed generation of a sequence of size " << input_size
             << "." << endl;
        cout << "sequence generation required " << max_time << " seconds."
             << endl;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    get_timer();

    // At this point every processor has a local list of random double precision
    // numbers distributed evenly over the interval (0,1).   The local list of
    // numbers is stored in the array local_numbers, where there are
    // local_input_size numbers in the array.  It is your task to sort these
    // numbers using the enumeration provided by MPI_Rank().

    //**************************************************************************/
    // Insert your parallel sort here, I've included the bitonic sort
    // for a simple example.
    //**************************************************************************/

    local_numbers =
        parallel_quick_sort(local_numbers, local_input_size, MPI_COMM_WORLD);

    double par_sort_time = get_timer();
    // Find out how much time it took to sort.
    MPI_Reduce(&par_sort_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0,
               MPI_COMM_WORLD);
    if (myid == 0) {
        cout << "parallel sort time = " << max_time << endl;
    }

    // Check to see if sequence is actually sorted.
    check_sort(local_numbers, local_input_size);

    MPI_Finalize();
    return 0;
}
