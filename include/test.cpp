#include "ensmallen.hpp"
#include <armadillo>
#include <chrono>
#include "../tests/test_function_tools.hpp"

int main()
{
  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
  arma::mat data, testData, shuffledData;
  arma::Row<size_t> responses, testResponses, shuffledResponses;

  LogisticRegressionTestData(data, testData, shuffledData,
      responses, testResponses, shuffledResponses);
  ens::test::LogisticRegression<> lr(shuffledData, shuffledResponses, 0.5);

  ens::AdaGrad ada(0.99, 32, 1e-8, 5000000, 1e-9, true);
  arma::mat coordinates = lr.GetInitialPoint();
  adagrad.Optimize(lr, coordinates);

  // Ensure that the error is close to zero.
  const double acc = lr.ComputeAccuracy(data, responses, coordinates);
  const double testAcc = lr.ComputeAccuracy(testData, testResponses,
      coordinates);
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  std::cout << "Time difference (sec) = " << (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) /1000000.0 <<std::endl;
}