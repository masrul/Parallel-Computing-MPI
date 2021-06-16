#include "game.h"
#include "utilities.h"
// Standard Includes for MPI, C and OS calls
#include <mpi.h>

// C++ standard I/O and library includes
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

// define some variables
#define server 0
#define chunk_size 8
#define work_avail 100     /* This tag means there is useful work to do */
#define terminate 101      /* This tag means no work, so terminate */
#define work_need 200      /* Client wants work */
#define solution_found 201 /* Client wants to send solutions */
#define client_done 202

// C++ stadard library using statements
using std::cerr;
using std::cout;
using std::endl;

using std::string;
using std::vector;

using std::ifstream;
using std::ios;
using std::ofstream;

void Server(int argc, char* argv[], int num_clients) {

    // Check to make sure the server can run
    if (argc != 3) {
        cerr << "two arguments please!" << endl;
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    // Input case filename
    ifstream input(argv[1], ios::in);

    // Output case filename
    ofstream output(argv[2], ios::out);

    // int count = 0 ;
    int NUM_GAMES = 0;
    
    // get the number of games from the input file
    input >> NUM_GAMES;

    unsigned char buf[NUM_GAMES * IDIM * JDIM];
    for (int i = 0; i < NUM_GAMES; ++i) { // for each game in file...
        string input_string;
        input >> input_string;
        if (input_string.size() != IDIM * JDIM) {
            cerr << "something wrong in input file format!" << endl;
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
        // read in the initial game state from file
        for (int j = 0; j < IDIM * JDIM; ++j) {
            buf[i * (IDIM * JDIM) + j] = input_string[j];
        }
    }

    int count = 0;
    int index = 0;
    int jobs = 0;
    unsigned char dummy[10000];
    // int msg_size;
    int Exist;
    MPI_Status stat;
    MPI_Status stat_temp;
    MPI_Request request;
    int terminate_send = 0;
    int client_end = 0;
    bool small_work = false;
    int msg_size = 100000;


    while (jobs < NUM_GAMES || client_end < num_clients) {
        MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &Exist, &stat);
        while (Exist) {
            int Source = stat.MPI_SOURCE;
            unsigned char result_buf[msg_size];
            MPI_Recv(&result_buf, msg_size, MPI_UNSIGNED_CHAR, Source,
                     MPI_ANY_TAG, MPI_COMM_WORLD, &stat);
            int Tag_Type = stat.MPI_TAG;
            if (Tag_Type == work_need) {
                if (NUM_GAMES == jobs) {
                    MPI_Isend(&dummy, 1, MPI_UNSIGNED_CHAR, Source, terminate,
                              MPI_COMM_WORLD, &request);
                } else if ((NUM_GAMES - jobs) < chunk_size) {
                    MPI_Isend(&dummy, 1, MPI_UNSIGNED_CHAR, Source, terminate,
                              MPI_COMM_WORLD, &request);
                } else {
                    MPI_Isend(&buf[jobs * (IDIM * JDIM)],
                              chunk_size * IDIM * JDIM, MPI_UNSIGNED_CHAR,
                              Source, work_avail, MPI_COMM_WORLD, &request);
                    jobs = jobs + chunk_size;
                }
            } else if (Tag_Type == solution_found) {
                output << result_buf << endl;
                ++count;
            } else {
                ++client_end;
            }
            MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &Exist,
                       &stat);
        }

        // Let server do some useful work
        if (jobs < NUM_GAMES) {
            unsigned char single_buf[IDIM * JDIM];
            for (int j = 0; j < IDIM * JDIM; ++j) {
                single_buf[j] = buf[jobs * IDIM * JDIM + j];
            }
            game_state game_board;
            game_board.Init(single_buf);

            move solution[IDIM * JDIM];
            int size = 0;
            bool found = depthFirstSearch(game_board, size, solution);

            if (found) {
                ++count;
            }

            ++jobs;
        }
    }

    cout << "found " << count << " solutions" << endl;
}

// Put the code for the client here
void Client(int ID) {
    unsigned char dummy;
    unsigned char buf[chunk_size * IDIM * JDIM];
    int solve = 0;
    bool loop = true;
    int Exist;
    unsigned char single_buf[IDIM * JDIM];
    while (loop) {
        MPI_Status stat;
        MPI_Send(&dummy, 1, MPI_UNSIGNED_CHAR, server, work_need,
                 MPI_COMM_WORLD);

        MPI_Iprobe(server, MPI_ANY_TAG, MPI_COMM_WORLD, &Exist, &stat);
        if (Exist) {
            MPI_Recv(&buf, chunk_size * IDIM * JDIM, MPI_UNSIGNED_CHAR, server,
                     MPI_ANY_TAG, MPI_COMM_WORLD, &stat);
            int Tag_Type = stat.MPI_TAG;
            if (Tag_Type == work_avail) {
                for (int i = 0; i < chunk_size; ++i) {
                    for (int j = 0; j < IDIM * JDIM; ++j) {
                        single_buf[j] = buf[i * IDIM * JDIM + j];
                    }
                    game_state game_board;
                    game_board.Init(single_buf);

                    move solution[IDIM * JDIM];
                    int size = 0;
                    bool found = depthFirstSearch(game_board, size, solution);

                    if (found) {
                        std::ostringstream internal_buf;
                        game_state s;
                        s.Init(single_buf);
                        s.Print(internal_buf);
                        for (int i = 0; i < size; ++i) {
                            s.makeMove(solution[i]);
                            internal_buf << "-->" << endl;
                            s.Print(internal_buf);
                        }
                        std::string result_buf = internal_buf.str();
                        result_buf.c_str();

                        MPI_Send(&result_buf, result_buf.size(),
                                 MPI_UNSIGNED_CHAR, server, solution_found,
                                 MPI_COMM_WORLD);
                    }
                }

            } else {
                loop = false;
            }
        }
    }
    MPI_Send(&dummy, 1, MPI_UNSIGNED_CHAR, server, client_done, MPI_COMM_WORLD);
}

int main(int argc, char* argv[]) {
    chopsigs_();

    // All MPI programs must call this function
    MPI_Init(&argc, &argv);

    int myId;
    int numProcessors;

    /* Get the number of processors and my processor identification */
    MPI_Comm_size(MPI_COMM_WORLD, &numProcessors);
    MPI_Comm_rank(MPI_COMM_WORLD, &myId);

    if (myId == 0) {
        // Processor 0 runs the server code
        get_timer(); // zero the timer
        Server(argc, argv, numProcessors - 1);
        // Measure the running time of the server
        printf("Num proce: %d", numProcessors);
        cout << "execution time = " << get_timer() << " seconds." << endl;
    } else {
        // all other processors run the client code.
        Client(myId);
    }

    // All MPI programs must call this before exiting
    MPI_Finalize();
}
