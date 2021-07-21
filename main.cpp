#include <iostream>
#include <Eigen>
#include <fstream>
#include <vector>


using namespace Eigen;

// std::vector<Agent> genAgents(int n);
// void rollout(std::vector<Agent> &population, Vector4d payoffX, Vector4d payoffY, double threshhold, int iperdouble=10);
// void writePopulation(std::vector<Agent> pop, int iteration);
// Matrix4d calcStationary(Matrix4d m, int x=12);
// void mergeSort(double a[], int begin, int end);
// void merge(double a[], int const left, int const mid, int const right);

class Agent {
    public:
    Agent(Vector4d s) {
        strategy = s;
        score = 0;
        age = 0;
    };

    Vector4d getStrategy() {
        return strategy;
    };

    void increaseScore(double value) {
        score += value;
    };

    void resetScore() {
        score = 0;
    };

    void incrementAge() {
        age++;
    };

    double getScore() {
        return score;
    };

    void display() {
        std::cout << strategy.transpose() << std::endl;
    };

    int getAge() {
        return age;
    };

    private:
    Vector4d strategy;
    double score;
    int age;
};

Matrix4d calcStationary(Matrix4d m, int x=12) {
    for (int i = 0; i++; i < x) {
        m *= m; // Finds the limiting stationary distribution
    }
    return m;
};

void merge(double a[], int const left, int const mid, int const right) {
    int const a1 = mid-left+1;
    int const a2 = right-mid;
    double* la = new double[a1]; 
    double* ra = new double[a2];

    // Split array between left and right
    for (int i = 0; i < a1; i++) {
        la[i] = a[left+i]; // Copy left+1 to mid
    }
    for (int j = 0; j < a2; j++) {
        ra[j] = a[mid+1+j]; // Copy mid+1 to right
    }

    int ai1 = 0;
    int ai2 = 0;
    int aim = left;

    // Merge into a single array, sorting every two elements (small to big)
    while (ai1 < a1 && ai2 < a2) {
        if (la[ai1] <= ra[ai2]) {
            a[aim] = la[ai1];
            ai1++;
        } else {
            a[aim] = ra[ai2];
            ai2++;
        }
        aim++;
    }
    
    // Remaining elements of la are added (that were too big)
    while (ai1 < a1) {
        a[aim] = la[ai1];
        ai1++;
        aim++;
    }

    // Remaining elements of ra are added (that were too small)
    while (ai2 < a2) {
        a[aim] = ra[ai2];
        ai2++;
        aim++;
    }
};

void mergeSort(double a[], int const begin, int const end) {
    if (begin >= end)
        return;
    int mid = begin + (end-begin)/2; // Finds the midpoint
    // Reduce the problem to a set of smaller arrays
    mergeSort(a, begin, mid);
    mergeSort(a, mid+1, end);
    // Once all smaller arrays have been sorted and merged, merge the largest array
    merge(a, begin, mid, end);
};

std::vector<Agent> genAgents(int n) {
    // This generates a matrix size n^4 * 4
    // This assumes equal strategic distribution
    std::vector<Agent> agents;
    VectorXd space = VectorXd::LinSpaced(n, 1/double(n), 1-1/(double)n);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                for (int l = 0; l < n; l++) {
                    Vector4d v;
                    v << space(i), space(j), space(k), space(l); 
                    agents.push_back(Agent(v)); 
                }
            }
        }
    }

//    for (int i = 0; i < pow(n,4); i+=(int)(pow(n,4)/100)) {
//        std::cout << "Strategy sample: \n" << strategies.row(i) << std::endl;
//    }

    return agents;
};

void writePopulation(std::vector<Agent> pop, int iteration) {
    std::ofstream file;
    file.open("record2", std::ios_base::app);
    file << "Iteration " << iteration << "\n";
    for (auto &it : pop) {
        file << it.getStrategy().transpose() << "\n";
    }
    file.close();
};

void rollout(std::vector<Agent> &population, Vector4d payoffX, Vector4d payoffY, double threshhold, int iperdouble=10) {
    int pop = population.size();
    for (auto &i : population) {
        i.resetScore();
    }
    for (int i = 0; i < pop; i++) {
        Vector4d px = population[i].getStrategy();
        Array44d mx;
        mx << px(0), px(1), px(2), px(3),
              px(0), px(1), px(2), px(3),
              1-px(0), 1-px(1), 1-px(2), 1-px(3),
              1-px(0), 1-px(1), 1-px(2), 1-px(3);
        mx << px(0), px(0), 1-px(0), 1-px(0),
              px(1), px(1), 1-px(1), 1-px(1),
              px(2), px(2), 1-px(2), 1-px(2),
              px(3), px(3), 1-px(3), 1-px(3);
        for (int j = i; j < pop; j++) {
            Vector4d py = population[j].getStrategy();
            Array44d my;
            my << py(0), 1-py(0), py(0), 1-py(0),
                  py(2), 1-py(2), py(2), 1-py(2),
                  py(1), 1-py(1), py(1), 1-py(1),
                  py(3), 1-py(3), py(3), 1-py(3);
            Matrix4d m = (mx*my).matrix();
//            assert(m*Vector4d::Ones() == Vector4d::Ones());
            RowVector4d mStat = calcStationary(m).row(0);
            population[i].increaseScore(mStat.dot(payoffX));
            if (i != j)
                population[j].increaseScore(mStat.dot(payoffY));
        }
        population[i].incrementAge();
    }

    double* scoresArray = new double[pop];
    double T = 0;
    for (int i = 0; i < pop; i++) {
        scoresArray[i] = population[i].getScore();
        T += scoresArray[i]/(double)pop;
    }
    mergeSort(scoresArray, 0, pop-1);
    double minScore = scoresArray[(int)(threshhold*pop)];
    std::cout << "Minimum score: " << minScore/(double)pop << std::endl;
    std::cout << "Average score: " << T/(double)pop << std::endl;
    std::cout << "Median score: " << scoresArray[(int)((double)pop/2)]/(double)pop << std::endl;
    std::cout << "Highest score: " << scoresArray[pop-1]/(double)pop << std::endl; 
    for (int i = pop-1; i >= 0; --i) {
        if (population[i].getScore() < minScore) {
            population.erase(population.begin()+i);
        }
        else if (population[i].getAge() % iperdouble == 0 && population[i].getAge() != 0)
            population.push_back(Agent(population[i].getStrategy())); 
    }
    std::cout << "Population change: " << (int)((int)population.size()-pop) << std::endl;
    std::cout << "Population size: " << population.size() << std::endl;
};

void tournament(int n, Vector4d payoffs, double threshhold) {
    for (int i = 0; i < 3; i++) {
        assert(payoffs(i) > payoffs(i+1));
    }
    Vector4d payoffX;
    Vector4d payoffY;
    payoffX << payoffs(1), payoffs(3), payoffs(0), payoffs(2); // CC, CD, DC, DD 
    payoffY << payoffs(1), payoffs(0), payoffs(3), payoffs(2); // CC, DC, CD, DD (from X's perspective)
    std::cout << "Using payoffs:\n" << payoffX << "\n" << payoffY << "\n" << std::endl;
    std::vector<Agent> population = genAgents(n);
    std::cout << "Finished generating strategies." << std::endl;
    int iteration = 0;
    while (iteration < 50) {
        std::cout << "Beginning iteration: " << iteration << std::endl;
        rollout(population, payoffX, payoffY, threshhold);
        writePopulation(population, iteration);
        iteration++;
    }
    std::cout << "Winners: \n" << std::endl;
    for (auto &it : population) {
        it.display();
    }
};

int main() {
    Vector4d payoffs;
    payoffs << 5, 3, 1, 0;
    tournament(6, payoffs, 0.1);
    return 0;
};
