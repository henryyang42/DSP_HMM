#include <cassert>
#include <cmath>
#include "hmm.h"
#define T 50
#define N_STATE 6
#define N_SEQ 10000
int obvs[N_SEQ][T], obv_N;
double alpha[T][N_STATE], beta[T][N_STATE];
double r[T][N_STATE], xi[T][N_STATE][N_STATE];
double forward(HMM& hmm, int obv[]) {
    for (int t = 0; t < T; t++)
        for (int j = 0; j < hmm.state_num; j++) {
            if (t == 0)
                alpha[t][j] = hmm.initial[j] * hmm.observation[obv[t]][j];
            else {
                alpha[t][j] = 0;
                for (int i = 0; i < hmm.state_num; i++)
                    alpha[t][j] += alpha[t - 1][i] * hmm.transition[i][j];
                alpha[t][j] *= hmm.observation[obv[t]][j];
            }
        }
    return 0;
}

double backword(HMM& hmm, int obv[]) {
    for (int t = T - 1; t >= 0; t--)
        for (int i = 0; i < hmm.state_num; i++) {
            if (t == T - 1)
                beta[t][i] = 1.0;
            else {
                beta[t][i] = 0;
                for (int j = 0; j < hmm.state_num; j++)
                    beta[t][i] += beta[t + 1][j] * hmm.transition[i][j] *
                                  hmm.observation[obv[t + 1]][j];
            }
        }
    return 0;
}

int main(int argc, char* argv[]) {
    int iter = atoi(argv[1]);
    char* model_init = argv[2];
    char* seq_model = argv[3];
    char* model = argv[4];
    HMM hmm_initial;
    loadHMM(&hmm_initial, model_init);
    FILE* f_seq_model = fopen(seq_model, "r");
    char s[100];

    while (fscanf(f_seq_model, "%s", s) > 0) {
        for (int i = 0; i < strlen(s); i++) obvs[obv_N][i] = s[i] - 'A';
        obv_N++;
    }
    printf("train_N=%d\n", obv_N);
    printf("iter=%d\n", iter);
    for (int it = 0; it < iter; it++) {
        double initial[MAX_STATE] = {};
        double transition[MAX_STATE][MAX_STATE] = {};
        double observation[MAX_OBSERV][MAX_STATE] = {};
        for (int obv_id = 0; obv_id < obv_N; obv_id++) {
            forward(hmm_initial, obvs[obv_id]);
            backword(hmm_initial, obvs[obv_id]);
            // make r
            for (int t = 0; t < T; t++) {
                double p = 0;
                for (int i = 0; i < hmm_initial.state_num; i++)
                    p += alpha[t][i] * beta[t][i];
                assert(p != 0);

                for (int i = 0; i < hmm_initial.state_num; i++)
                    r[t][i] = alpha[t][i] * beta[t][i] / p;
            }
            // make xi
            for (int t = 0; t < T - 1; t++) {
                double p = 0;
                for (int i = 0; i < hmm_initial.state_num; i++)
                    for (int j = 0; j < hmm_initial.state_num; j++)
                        p += alpha[t][i] * hmm_initial.transition[i][j] *
                             hmm_initial.observation[obvs[obv_id][t + 1]][j] *
                             beta[t + 1][j];
                assert(p != 0);
                for (int i = 0; i < hmm_initial.state_num; i++)
                    for (int j = 0; j < hmm_initial.state_num; j++)
                        xi[t][i][j] =
                            alpha[t][i] * hmm_initial.transition[i][j] *
                            hmm_initial.observation[obvs[obv_id][t + 1]][j] *
                            beta[t + 1][j] / p;
            }
            // local update
            for (int i = 0; i < hmm_initial.state_num; i++) {
                initial[i] += r[0][i];
            }

            for (int i = 0; i < hmm_initial.state_num; i++) {
                double p = 0;
                for (int t = 0; t < T - 1; t++) p += r[t][i];
                assert(p != 0);
                for (int j = 0; j < hmm_initial.state_num; j++) {
                    for (int t = 0; t < T - 1; t++)
                        transition[i][j] += xi[t][i][j] / p;
                }
            }

            for (int i = 0; i < hmm_initial.state_num; i++) {
                double p2 = 0, p[MAX_OBSERV] = {};
                for (int t = 0; t < T; ++t) {
                    p[obvs[obv_id][t]] += r[t][i];
                    p2 += r[t][i];
                }
                assert(p2 != 0);

                for (int o = 0; o < hmm_initial.observ_num; o++)
                    observation[o][i] += p[o] / p2;
            }
        }
        // Update
        for (int i = 0; i < hmm_initial.state_num; i++) {
            hmm_initial.initial[i] = initial[i] / obv_N;
        }

        for (int i = 0; i < hmm_initial.state_num; i++)
            for (int j = 0; j < hmm_initial.state_num; j++)
                hmm_initial.transition[i][j] = transition[i][j] / obv_N;

        for (int o = 0; o < hmm_initial.observ_num; o++)
            for (int i = 0; i < hmm_initial.state_num; i++)
                hmm_initial.observation[o][i] = observation[o][i] / obv_N;
    }
    dumpHMM(fopen(model, "w"), &hmm_initial);
    return 0;
}