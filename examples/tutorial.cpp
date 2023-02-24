#include "../DAFMM2D.cpp"
#include <boost/math/special_functions/bessel.hpp>
double timeEntry;

double besselJ(int n, double x) {
	if (n >= 0) {
		double temp = boost::math::cyl_bessel_j(double(n), x);
		return temp;
	}
	else {
		double temp = boost::math::cyl_bessel_j(double(-n), x);
		if (-n%2 == 0)
			return temp;
		else
			return -temp;
	}
}

double besselY(int n, double x) {
	if (n >= 0) {
		long double temp = boost::math::cyl_neumann(double(n), x);
		return temp;
	}
	else {
		long double temp = boost::math::cyl_neumann(double(-n), x);
		if (-n%2 == 0)
			return temp;
		else
			return -temp;
	}
}

kernel_dtype FMM_Matrix::getMatrixEntry(const unsigned i, const unsigned j) {
	double start		=	omp_get_wtime();

	pts2D ri = particles_X[i];//particles_X is a member of base class FMM_Matrix
	pts2D rj = particles_X[j];//particles_X is a member of base class FMM_Matrix
	double R2 = (ri.x-rj.x)*(ri.x-rj.x) + (ri.y-rj.y)*(ri.y-rj.y);
	double R = sqrt(R2);

	kernel_dtype out;
	if (kappa*R ==0) {
	// if (kappa*R < 1e-6) {
		out = R;
	}
	else {
		out = besselJ(0, kappa*R) + I*besselY(0, kappa*R);
		// kernel_dtype out = besselY(0, kappa*R);
	}
	double end		=	omp_get_wtime();
	timeEntry += end-start;
	return out;
}

int main(int argc, char* argv[]) {
	int nCones_LFR;
	int nChebNodes;
	double L;
  int TOL_POW;
	int yes2DFMM;
	if(argc < 4)
	{
			std::cout << "All arguments weren't passed to executable!" << std::endl;
			std::cout << "Using Default Arguments:" << std::endl;
			nCones_LFR	=	16;
			nChebNodes	=	6;
			L						=	1.0;
			kappa				= 100.0;
		  TOL_POW			= 8;
			yes2DFMM		=	1;
	}

	else
	{
		nCones_LFR	=	atoi(argv[1]);
		nChebNodes	=	atoi(argv[2]);
		L						=	atof(argv[3]);
		kappa				= atof(argv[4]);
		TOL_POW			= atoi(argv[5]);
		yes2DFMM		=	atoi(argv[6]);
	}

	timeEntry = 0;
  inputsToDFMM inputs;
  inputs.nCones_LFR = nCones_LFR;
  inputs.nChebNodes = nChebNodes;
  inputs.L = L;
  inputs.yes2DFMM = yes2DFMM;
  inputs.TOL_POW = TOL_POW;
  Vec Phi;
	double start, end;

	start		=	omp_get_wtime();
	DAFMM2D *dafmm2d = new DAFMM2D(inputs);
	end		=	omp_get_wtime();
	double timeInitialise =	(end-start);
	std::cout << "========================= Initialisation Time =========================" << std::endl;
	std::cout << "Time for initialising DAFMM    :" << timeInitialise << std::endl;

  int N = dafmm2d->K->N;
  Vec b = Vec::Random(N);

  std::cout << "========================= Problem Parameters =========================" << std::endl;
  std::cout << "Matrix Size                        :" << N << std::endl;
  std::cout << "Tolerance                          :" << pow(10,-TOL_POW) << std::endl << std::endl;

  start		=	omp_get_wtime();
	dafmm2d->assemble();
	end		=	omp_get_wtime();
	double timeAssemble =	(end-start);
  std::cout << "========================= Assembly Time =========================" << std::endl;
  std::cout << "Time for assemble in DAFMM form    :" << timeAssemble << std::endl;
	std::cout << "Time for assemble in DAFMM form without matrixEntry time  :" << timeAssemble-timeEntry << std::endl;

	Vec DAFMM_Ab;
  start		=	omp_get_wtime();
	dafmm2d->MatVecProduct(b, DAFMM_Ab);
  end		=	omp_get_wtime();
  double timeMatVecProduct =	(end-start);
  std::cout << "========================= Matrix-Vector Multiplication =========================" << std::endl;
  std::cout << "Time for MatVec in DAFMM form      :" << timeMatVecProduct << std::endl;

	// exit(0);
	/////////////////////////////////
  start		=	omp_get_wtime();
	// Vec true_Ab = Afull*b;
	Vec true_Ab = Vec::Zero(N);
	#pragma omp parallel for
	for (size_t i = 0; i < N; i++) {
		// #pragma omp parallel for
		for (size_t j = 0; j < N; j++) {
			true_Ab(i) += dafmm2d->K->getMatrixEntry(i,j)*b(j);
		}
	}
  end		=	omp_get_wtime();
  double exact_time =	(end-start);
  std::cout << "Time for direct MatVec             :" << exact_time << std::endl;
  std::cout << "Magnitude of Speed-Up              :" << (exact_time / timeMatVecProduct) << std::endl;

	double err = (true_Ab - DAFMM_Ab).norm()/true_Ab.norm();
  std::cout << "Error in the solution is           :" << err << std::endl << std::endl;
	delete dafmm2d;
}
