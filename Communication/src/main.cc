#include "utilities.h"
#include <fstream>
#include <iostream>
#include <mpi.h>
#include <sstream>
#include <stdlib.h>
#include <string>
#define ecube_tag 100
#define hypercube_alltoallprsnl
#define naive_nonblocking_alltoall

using namespace std;
int numprocs, myid;

/******************************************************************************
 * evaluate 2^i                                                                *
 ******************************************************************************/
inline unsigned int pow2(unsigned int i) { return 1 << i; }

/******************************************************************************
 * evaluate ceil(log2(i))                                                      *
 ******************************************************************************/
inline unsigned int log2(unsigned int i) {
    i--;
    unsigned int log = 1;
    for (i >>= 1; i != 0; i >>= 1)
        log++;
    return log;
}

/******************************************************************************
 * This function should implement the All to All Broadcast.  The value for each*
 * processor is given by the argument send_value.  The recv_buffer argument is *
 * an array of size p that will store the values that each processor transmits.*
 * See Program 4.7 page 162 of the text.                                       *
 ******************************************************************************/

void AllToAll(int send_value[], int recv_buffer[], int size, MPI_Comm comm) {
#ifdef naive_nonblocking_alltoall
    int d = log2(numprocs);
    MPI_Status status[2 * numprocs - 2];
    MPI_Request request[2 * numprocs - 2];
    // Align my own data into recv buffer
    for (int i = 0; i < size; ++i) {
        recv_buffer[myid * size + i] = send_value[i];
    }
    int index = 0;

    for (int i = 0; i < numprocs; ++i) {
        if (myid == i)
            continue;
        int partner = i;
        MPI_Irecv(&recv_buffer[partner * size], size, MPI_INT, partner,
                  ecube_tag, comm, &request[index]);
        ++index;
        MPI_Isend(&recv_buffer[myid * size], size, MPI_INT, partner, ecube_tag,
                  comm, &request[index]);
        ++index;
    }
    MPI_Waitall(2 * numprocs - 2, request, status);
#endif

#ifdef recursive_doubling_alltoall
    MPI_Status status;
    int d = log2(numprocs);
    int mask = pow2(d - 1); // to toggle Most significant bit
    int twinid;
    bool twin = true;

    // If i have a twin, then it will be > #proc-1
    if ((myid ^ mask) > numprocs - 1) {
        twinid = myid ^ mask;
    } else {
        twin = false; // Does not exist
    }

    // Align my own data into recv buffer
    for (int i = 0; i < size; ++i) {
        recv_buffer[myid * size + i] = send_value[i];
    }

    // Hypercube loop start
    for (int i = 0; i < d; ++i) {
        int partner = myid ^ pow2(i);
        if (twin) {
            if (twinid == partner)
                goto twinpart;
        }
        int send_index = myid >> i;
        send_index <<= i;
        int recv_index = partner >> i;
        recv_index <<= i;

        // correction of message block
        // Ther are p-1 blocks of message, this number cann't be excced
        int send_size;
        bool send = true;
        if (send_index > numprocs - 1)
            send = false;
        if (send_index + pow2(i) > numprocs - 1) {
            send_size = size * (numprocs - send_index);
        } else {
            send_size = size * pow2(i);
        }
        int recv_size;
        bool recv = true;
        if (recv_index > numprocs - 1)
            recv = false;
        if (recv_index + pow2(i) > numprocs - 1) {
            recv_size = size * (numprocs - recv_index);
        } else {
            recv_size = size * pow2(i);
        }

        if (myid < partner) {
            if (partner > (numprocs - 1))
                partner = partner ^ mask;
            if (send)
                MPI_Send(&recv_buffer[send_index * size], send_size, MPI_INT,
                         partner, 100, comm);
            if (recv)
                MPI_Recv(&recv_buffer[recv_index * size], recv_size, MPI_INT,
                         partner, 100, comm, &status);

        } else if (myid > partner) {
            if (recv)
                MPI_Recv(&recv_buffer[recv_index * size], recv_size, MPI_INT,
                         partner, 100, comm, &status);
            if (send)
                MPI_Send(&recv_buffer[send_index * size], send_size, MPI_INT,
                         partner, 100, comm);
        }

    // Check if i have twin
    // Let's do my twin's work if exists
    twinpart:
        if (twin) {
            int partner = twinid ^ pow2(i);
            if (myid == partner)
                continue;

            int send_index = twinid >> i;
            send_index <<= i;

            int recv_index = partner >> i;
            recv_index <<= i;
            int send_size;
            bool send = true;
            if (send_index > numprocs - 1)
                send = false;
            if (send_index + pow2(i) > numprocs - 1) {
                send_size = size * (numprocs - send_index);
            } else {
                send_size = size * pow2(i);
            }
            int recv_size;
            bool recv = true;
            if (recv_index > numprocs - 1)
                recv = false;
            if (recv_index + pow2(i) > numprocs - 1) {
                recv_size = size * (numprocs - recv_index);
            } else {
                recv_size = size * pow2(i);
            }

            if (twinid < partner) {
                if (partner > numprocs - 1)
                    partner = partner ^ mask;
                if (send)
                    MPI_Ssend(&recv_buffer[send_index * size], send_size,
                              MPI_INT, partner, 100, comm);
                if (recv)
                    MPI_Recv(&recv_buffer[recv_index * size], recv_size,
                             MPI_INT, partner, 100, comm, &status);
            } else if (twinid > partner) {
                if (partner > numprocs - 1)
                    partner = partner ^ mask;
                if (recv)
                    MPI_Recv(&recv_buffer[recv_index * size], recv_size,
                             MPI_INT, partner, 100, comm, &status);
                if (send)
                    MPI_Ssend(&recv_buffer[send_index * size], send_size,
                              MPI_INT, partner, 100, comm);
            }
        }
    }

#endif

#ifdef ring_alltoall
    MPI_Status status;

    // Align my own data into recv buffer
    for (int i = 0; i < size; ++i) {
        recv_buffer[myid * size + i] = send_value[i];
    }

    int left_partner = (myid + numprocs - 1) % numprocs;
    int right_partner = (myid + 1) % numprocs;

    // Set send and recv iteration for first iteration
    int recv_index = left_partner;
    int send_index = myid;

    for (int i = 1; i < numprocs; ++i) {
        if (myid % 2 == 0) {
            MPI_Send(&recv_buffer[send_index * size], size, MPI_INT,
                     right_partner, 100, comm);
            MPI_Recv(&recv_buffer[recv_index * size], size, MPI_INT,
                     left_partner, 100, comm, &status);
        } else {
            MPI_Recv(&recv_buffer[recv_index * size], size, MPI_INT,
                     left_partner, 100, comm, &status);
            MPI_Send(&recv_buffer[send_index * size], size, MPI_INT,
                     right_partner, 100, comm);
        }

        // Set send and recv index for next iteration
        send_index = recv_index; // I  am sending, whatever i just received
        recv_index = (recv_index + numprocs - 1) % numprocs;
    }

#endif
}

/******************************************************************************
 * This function should implement the All to All Personalized Broadcast.       *
 * A value destined for each processor is given by the argument array          *
 * send_buffer of size p.  The recv_buffer argument is an array of size p      *
 * that will store the values that each processor transmits.                   *
 * See pages 175-179 in the text.                                              *
 ******************************************************************************/

void AllToAllPersonalized(int send_buffer[], int recv_buffer[], int size,
                          MPI_Comm comm) {
    // MPI_Alltoall(send_buffer,size,MPI_INT,recv_buffer,size,MPI_INT,comm) ;
#ifdef ecube_alltoallprsnl
    int numprocs, myid;
    MPI_Comm_size(comm, &numprocs);
    MPI_Comm_rank(comm, &myid);
    int d = log2(numprocs);
    MPI_Status status;

    // Align my own data into recv buffer
    for (int i = 0; i < size; ++i) {
        recv_buffer[myid * size + i] = send_buffer[myid * size + i];
    }

    for (int i = 1; i < numprocs; ++i) {
        int partner = myid ^ i;
        if (myid < partner) {
            MPI_Send(&send_buffer[partner * size], size, MPI_INT, partner,
                     ecube_tag, comm);
            MPI_Recv(&recv_buffer[partner * size], size, MPI_INT, partner,
                     ecube_tag, comm, &status);
        } else {
            MPI_Recv(&recv_buffer[partner * size], size, MPI_INT, partner,
                     ecube_tag, comm, &status);
            MPI_Send(&send_buffer[partner * size], size, MPI_INT, partner,
                     ecube_tag, comm);
        }
    }
#endif

#ifdef hypercube_alltoallprsnl
    int d = log2(numprocs);
    MPI_Status status;

    // Align my own data into recv buffer
    for (int i = 0; i < size; ++i) {
        recv_buffer[myid * size + i] = send_buffer[myid * size + i];
    }
    int count = numprocs / 2;
    const int max_size = pow2(18);
    int temp_send_buffer[count * max_size];
    int temp_recv_buffer[count * max_size];

    for (int i = 0; i < d; ++i) {
        int partner = myid ^ pow2(i);
        int myithbit = (myid >> i) & 1;
        int checkbit;
        if (myithbit == 1) {
            checkbit = 0;
        } else {
            checkbit = 1;
        }

        if (myid < partner) {
            // I am  going sending first and receiving later to avoid deadlock
            // Let's pack sending data into temp_send_buffer as, data is not
            // memory contiguous
            // I will send the data location whose ith bit is 1
            int index = 0;
            for (int j = 0; j < numprocs; ++j) {
                if ((j >> i) & 1 == checkbit) {
                    temp_send_buffer [index * size:index * size + size - 1] =
                    send_buffer [j * size:j * size + size - 1];
                    ++index;
                }
            }
            MPI_Send(temp_send_buffer, count * size, MPI_INT, partner, 100,
                     comm);
            MPI_Recv(temp_recv_buffer, count * size, MPI_INT, partner, 100,
                     comm, &status);
            // Let's unpack received data into my actual recv_buffer
            index = 0;
            for (int j = 0; j < numprocs; ++j) {
                if ((j >> i) & 1 == checkbit) {
                    recv_buffer [j * size:j * size + size - 1] =
                    temp_recv_buffer [index * size:index * size + size - 1];
                    ++index;
                }
            }
        } else {
            MPI_Recv(temp_recv_buffer, count * size, MPI_INT, partner, 100,
                     comm, &status);
            // Let's unpack received data into my actual recv_buffer
            int index = 0;
            for (int j = 0; j < numprocs; ++j) {
                if ((j >> i) & 1 == checkbit) {
                    recv_buffer [j * size:j * size + size - 1] =
                    temp_recv_buffer [index * size:index * size + size - 1];
                    ++index;
                }
            }
            // Let's pack sending data into temp_send_buffer as, data is not
            // memory contiguous
            index = 0;
            for (int j = 0; j < numprocs; ++j) {
                if ((j >> i) & 1 == checkbit) {
                    temp_send_buffer [index * size:index * size + size - 1] =
                    send_buffer [j * size:j * size + size - 1];
                    ++index;
                }
            }
            MPI_Send(temp_send_buffer, count * size, MPI_INT, partner, 100,
                     comm);
        }
    }
#endif

#ifdef naive_nonblocking_alltoallprsnl
    // Ref: Rajeev Thakur, William Gropp, ANL/ACM,-P1007-1102
    //
    int d = log2(numprocs);
    MPI_Status status[2 * numprocs - 2];
    MPI_Request request[2 * numprocs - 2];

    // Align my own data into recv buffer
    for (int i = 0; i < size; ++i) {
        recv_buffer[myid * size + i] = send_buffer[myid * size + i];
    }

    int index = 0;
    for (int i = 0; i < numprocs; ++i) {
        if (myid == i)
            continue;
        int partner = i;
        MPI_Irecv(&recv_buffer[partner * size], size, MPI_INT, partner,
                  ecube_tag, comm, &request[index]);
        ++index;
        MPI_Isend(&send_buffer[partner * size], size, MPI_INT, partner,
                  ecube_tag, comm, &request[index]);
        ++index;
    }
    MPI_Waitall(2 * numprocs - 2, request, status);

#endif

#ifdef naive_wraparound_alltoallprsnl
    // Ref: Rajeev Thakur, William Gropp, ANL/ACM,-P1007-1102
    MPI_Status status;

    // Align my own data into recv buffer
    for (int i = 0; i < size; ++i) {
        recv_buffer[myid * size + i] = send_buffer[myid * size + i];
    }

    for (int i = 1; i < numprocs; ++i) {
        int src = (myid - i + numprocs) % numprocs;
        int dest = (myid + i) % numprocs;
        MPI_Sendrecv(&send_buffer[dest * size], size, MPI_INT, dest, ecube_tag,
                     &recv_buffer[src * size], size, MPI_INT, src, ecube_tag,
                     comm, &status);
    }

#endif
}

int main(int argc, char** argv) {

    chopsigs_();

    double time_passes, max_time;
    /* Initialize MPI */
    MPI_Init(&argc, &argv);

    /* Get the number of processors and my processor identification */
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    int test_runs = 8000 / numprocs;
    if (argc == 2)
        test_runs = atoi(argv[1]);
    const int max_size = pow2(16);
    int* recv_buffer = new int[numprocs * max_size];
    int* send_buffer = new int[numprocs * max_size];

    if (0 == myid) {
        cout << "Starting " << numprocs << " processors."
             << " Testruns:  " << test_runs << endl;
    }

    /***************************************************************************/
    /* Check Timing for Single Node Broadcast emulating an alltoall broadcast */
    /***************************************************************************/

    MPI_Barrier(MPI_COMM_WORLD);

    /* We can't accurately measure short times so we must execute this operation
       many times to get accurate timing measurements.*/
    for (int l = 0; l <= 16; l += 4) {
        int msize = pow2(l);
        /* Every call to get_timer resets the stopwatch.  The next call to
           get_timer will return the amount of time since now */
        get_timer();
        for (int i = 0; i < test_runs; ++i) {
            /* Slow alltoall broadcast using p single node broadcasts */
            for (int p = 0; p < numprocs; ++p)
                recv_buffer[p] = 0;
            int send_info = myid + i * numprocs;
            for (int k = 0; k < msize; ++k)
                send_buffer[k] = send_info;
            AllToAll(send_buffer, recv_buffer, msize, MPI_COMM_WORLD);

            for (int p = 0; p < numprocs; ++p)
                if (recv_buffer[p * msize] != (p + i * numprocs))
                    cerr << "recv failed on processor " << myid
                         << " recv_buffer[" << p
                         << "] = " << recv_buffer[p * msize] << " should  be "
                         << p + i * numprocs << endl;
        }
        time_passes = get_timer();

        MPI_Reduce(&time_passes, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0,
                   MPI_COMM_WORLD);
        if (0 == myid)
            cout << "all to all broadcast for m=" << msize << " required "
                 << max_time / double(test_runs) << " seconds." << endl;
    }

    /***************************************************************************/
    /* Check Timing for All to All personalized Broadcast Algorithm */
    /***************************************************************************/

    MPI_Barrier(MPI_COMM_WORLD);

    for (int l = 0; l <= 14; l += 4) {
        int msize = pow2(l);
        /* Every call to get_timer resets the stopwatch.  The next call to
           get_timer will return the amount of time since now */
        get_timer();

        for (int i = 0; i < test_runs; ++i) {
            for (int p = 0; p < numprocs; ++p)
                for (int k = 0; k < msize; ++k)
                    recv_buffer[p * msize + k] = 0;
            int factor = (myid & 1 == 1) ? -1 : 1;
            for (int p = 0; p < numprocs; ++p)
                for (int k = 0; k < msize; ++k)
                    send_buffer[p * msize + k] =
                        myid * numprocs + p + i * myid * myid * factor;
            int send_info = myid + i * numprocs;

            AllToAllPersonalized(send_buffer, recv_buffer, msize,
                                 MPI_COMM_WORLD);

            for (int p = 0; p < numprocs; ++p) {
                int factor = (p & 1 == 1) ? -1 : 1;
                if (recv_buffer[p * msize] !=
                    (p * numprocs + myid + i * p * p * factor))
                    cout << "recv failed on processor " << myid
                         << " recv_buffer[" << p
                         << "] = " << recv_buffer[p * msize] << " should  be "
                         << p * numprocs + myid + i * p * p * factor << endl;
            }
        }

        time_passes = get_timer();

        MPI_Reduce(&time_passes, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0,
                   MPI_COMM_WORLD);
        if (0 == myid)
            cout << "all-to-all-personalized broadcast, m=" << msize
                 << " required " << max_time / double(test_runs) << " seconds."
                 << endl;
    }

    /* We're finished, so call MPI_Finalize() to clean things up */
    MPI_Finalize();
    return 0;
}
