#include <cmath>
#include "hmm.h"
#define T 50
double viterbi(HMM& hmm, int obv[]) {
    double p[T][6] = {};
    for (int t = 0; t < T; t++) {
        for (int i = 0; i < hmm.state_num; i++) {
            if (t == 0)
                p[t][i] = hmm.initial[i] * hmm.observation[obv[t]][i];
            else {
                for (int j = 0; j < hmm.state_num; j++)
                    p[t][j] = fmax(p[t][j], p[t - 1][i] * hmm.transition[i][j] *
                                                hmm.observation[obv[t]][j]);
            }
        }
    }
    double v = -1;
    for (int i = 0; i < hmm.state_num; i++)
        if (p[T - 1][i] > v) v = p[T - 1][i];
    return v;
}

int obvs[2500][T], obv_N, anss_N;
char anss[2500][20];

int main(int argc, char* argv[]) {
    FILE* f_seq_model = fopen("../testing_data1.txt", "r");
    FILE* f_ans = fopen("ans.txt", "w");
    FILE* f_anss = fopen("../testing_answer.txt", "r");
    char s[100];
    while (fscanf(f_seq_model, "%s", s) > 0) {
        for (int i = 0; i < strlen(s); i++) obvs[obv_N][i] = s[i] - 'A';
        obv_N++;
    }
    while (fscanf(f_anss, "%s", anss[anss_N++]) > 0)
        ;

    HMM hmms[5];
    double acc = 0;
    load_models("modellist.txt", hmms, 5);
    for (int i = 0; i < obv_N; i++) {
        int max_ = 0;
        double max_v = -1;
        for (int j = 0; j < 5; j++) {
            double v = viterbi(hmms[j], obvs[i]);
            if (v > max_v) {
                max_v = v;
                max_ = j;
            }
        }
        fprintf(f_ans, "%s\n", hmms[max_].model_name);
        if (strcmp(hmms[max_].model_name, anss[i]) == 0) acc += 1;
    }
    printf("acc=%.0lf (%lf%%)\n", acc, acc / obv_N * 100);
    return 0;
}
