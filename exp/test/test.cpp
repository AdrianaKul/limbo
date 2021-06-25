//| Copyright Inria May 2015
//| This project has received funding from the European Research Council (ERC) under
//| the European Union's Horizon 2020 research and innovation programme (grant
//| agreement No 637972) - see http://www.resibots.eu
//|
//| Contributor(s):
//|   - Jean-Baptiste Mouret (jean-baptiste.mouret@inria.fr)
//|   - Antoine Cully (antoinecully@gmail.com)
//|   - Konstantinos Chatzilygeroudis (konstantinos.chatzilygeroudis@inria.fr)
//|   - Federico Allocati (fede.allocati@gmail.com)
//|   - Vaios Papaspyros (b.papaspyros@gmail.com)
//|   - Roberto Rama (bertoski@gmail.com)
//|
//| This software is a computer library whose purpose is to optimize continuous,
//| black-box functions. It mainly implements Gaussian processes and Bayesian
//| optimization.
//| Main repository: http://github.com/resibots/limbo
//| Documentation: http://www.resibots.eu/limbo
//|
//| This software is governed by the CeCILL-C license under French law and
//| abiding by the rules of distribution of free software.  You can  use,
//| modify and/ or redistribute the software under the terms of the CeCILL-C
//| license as circulated by CEA, CNRS and INRIA at the following URL
//| "http://www.cecill.info".
//|
//| As a counterpart to the access to the source code and  rights to copy,
//| modify and redistribute granted by the license, users are provided only
//| with a limited warranty  and the software's author,  the holder of the
//| economic rights,  and the successive licensors  have only  limited
//| liability.
//|
//| In this respect, the user's attention is drawn to the risks associated
//| with loading,  using,  modifying and/or developing or reproducing the
//| software by the user in light of its specific status of free software,
//| that may mean  that it is complicated to manipulate,  and  that  also
//| therefore means  that it is reserved for developers  and  experienced
//| professionals having in-depth computer knowledge. Users are therefore
//| encouraged to load and test the software's suitability as regards their
//| requirements in conditions enabling the security of their systems and/or
//| data to be ensured and,  more generally, to use and operate it in the
//| same conditions as regards security.
//|
//| The fact that you are presently reading this means that you have had
//| knowledge of the CeCILL-C license and that you accept its terms.
//|
#include <fstream>
#include <limbo/kernel/exp.hpp>
#include <limbo/kernel/squared_exp_ard.hpp>
#include <limbo/mean/data.hpp>
#include <limbo/model/gp.hpp>
#include <limbo/model/gp/kernel_lf_opt.hpp>
#include <limbo/tools.hpp>
#include <limbo/tools/macros.hpp>

#include <limbo/serialize/text_archive.hpp>

// Random boost requirements
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_real_distribution.hpp>
#include <boost/random/binomial_distribution.hpp>
#include <boost/random/normal_distribution.hpp>

// this tutorials shows how to use a Gaussian process for regression

using namespace limbo;

struct Params
{
    struct kernel_exp
    {
        BO_PARAM(double, sigma_sq, 0.2);
        BO_PARAM(double, l, 5.1);
    };
    struct kernel : public defaults::kernel
    {
        BO_PARAM(double, noise, 0.01);
        // BO_PARAM(bool, optimize_noise, true);
    };
    struct kernel_squared_exp_ard : public defaults::kernel_squared_exp_ard
    {
    };
    struct opt_rprop : public defaults::opt_rprop
    {
    };
    struct mean_constant : public defaults::mean_constant
    {
        BO_PARAM(double, constant, 0);
    };
};

int main(int argc, char const *argv[])
{
    int dim_in = 2;
    int dim_out = 2;
    double range = 2.;
    double noise_level = 0.01;
    double conductivity = 3;

    Eigen::VectorXd param_in = Eigen::VectorXd(2);
    param_in << std::log(std::stod(argv[1])), std::log(std::sqrt(std::stod(argv[2])));

    std::cout << param_in.transpose() << " = param in \n";

    // our data (1-D inputs, 1-D outputs)
    std::vector<Eigen::VectorXd> samples;
    std::vector<Eigen::VectorXd> observations;

    // boost random
    boost::random::mt11213b gen; // generator
    // gen.seed(static_cast<unsigned int>(std::time(0)));
    boost::random::uniform_real_distribution<> dist(
        -range, range); // distribution uniform real
    boost::random::normal_distribution<> noise(
        0, noise_level); // distribution normal

    if (dim_in == 2 && dim_out == 1)
    {
        size_t N = 20;
        for (size_t i = 0; i < N; i++)
        {
            double s1 = dist(gen);
            Eigen::VectorXd s(2);
            // s << dist(gen), dist(gen);
            s << s1, s1 * conductivity;
            // s << s1, s1 * conductivity + noise(gen);
            // Eigen::VectorXd s = 2 * Eigen::VectorXd::Random(2);
            samples.push_back(s);
            Eigen::VectorXd o(1);
            o[0] = s[0];
            // o[0] = std::cos(s[0]);
            observations.push_back(o);
        }
    }
    if (dim_in == 2 && dim_out == 2)
    {
        size_t N = 100;
        for (size_t i = 0; i < N; i++)
        {
            double s1 = dist(gen);
            Eigen::VectorXd s(2);
            // s << s1, s1 * conductivity;
            s << s1, s1 * conductivity + noise(gen);
            // s << s1, s1 * (conductivity + noise(gen));
            samples.push_back(s);
            observations.push_back(s);
        }
    }

    // the type of the GP
    using Kernel_t = kernel::Exp<Params>;
    using Mean_t = mean::Constant<Params>;
    using GP_t = model::GP<Params, Kernel_t, Mean_t>;

    // 1-D inputs, 1-D outputs
    GP_t gp(dim_in, dim_out);
    gp.kernel_function().set_params(param_in);

    // compute the GP
    gp.compute(samples, observations);

    if (dim_in == 2 && dim_out == 1)
    {
        // write the predicted data in a file (e.g. to be plotted)
        std::ofstream ofs("gp.dat");
        for (int i = 0; i < 100; ++i)
        {
            Eigen::VectorXd v(2);
            v[0] = (i / 100.0) * 5.0 - 2.5;
            v[1] = 0.0;
            Eigen::VectorXd mu;
            double sigma;
            std::tie(mu, sigma) = gp.query(v);
            // an alternative (slower) is to query mu and sigma separately:
            //  double mu = gp.mu(v)[0]; // mu() returns a 1-D vector
            //  double s2 = gp.sigma(v);
            ofs << v.transpose() << " " << mu[0] << " " << sqrt(sigma) << std::endl;
        }
    }

    if (dim_in == 2 && dim_out == 2)
    {
        // write the predicted data in a file (e.g. to be plotted)
        std::ofstream ofs("gp.dat");
        for (int i = 0; i < 100; ++i)
        {
            Eigen::VectorXd v(2);
            v[0] = (i / 100.0) * 5.0 - 2.5;
            v[1] = 0.0;
            Eigen::VectorXd mu;
            double sigma;
            std::tie(mu, sigma) = gp.query(v);
            // an alternative (slower) is to query mu and sigma separately:
            //  double mu = gp.mu(v)[0]; // mu() returns a 1-D vector
            //  double s2 = gp.sigma(v);
            ofs << v[0] << " " << mu.transpose() << " " << sqrt(sigma) << std::endl;
        }
    }

    // gp surface
    if (dim_in == 2 && dim_out == 2)
    {
        // write the predicted data in a file (e.g. to be plotted)
        std::ofstream ofs_surface("gp_surface.dat");
        for (int i = 0; i < 20; ++i)
        {
            for (int j = 0; j < 20; ++j)
            {
                Eigen::VectorXd v(2);
                v[0] = (i / 20.0) * 5.0 - 2.5;
                v[1] = (j / 20.0) * 5.0 - 2.5;
                Eigen::VectorXd mu;
                double sigma;
                std::tie(mu, sigma) = gp.query(v);
                // an alternative (slower) is to query mu and sigma separately:
                //  double mu = gp.mu(v)[0]; // mu() returns a 1-D vector
                //  double s2 = gp.sigma(v);
                ofs_surface << v.transpose() << " " << mu.transpose() << " " << sqrt(sigma) << std::endl;
            }
        }
    }

    std::cout << "noise = " << gp.kernel_function().noise() << "\n";
    std::cout << "params = " << gp.kernel_function().params().transpose() << "\n";
    std::cout << "mean = " << gp.mean_function().h_params() << "\n";

    // an alternative is to optimize the hyper-parameters
    // in that case, we need a kernel with hyper-parameters that are designed to be optimized
    using Kernel2_t = kernel::SquaredExpARD<Params>;
    // using Mean_t = mean::Data<Params>;
    using GP2_t = model::GP<Params, Kernel2_t, Mean_t, model::gp::KernelLFOpt<Params>>;

    GP2_t gp_ard(dim_in, dim_out);
    // do not forget to call the optimization!
    gp_ard.compute(samples, observations, false);
    gp_ard.optimize_hyperparams();

    if (dim_in == 2 && dim_out == 1)
    {
        // write the predicted data in a file (e.g. to be plotted)
        std::ofstream ofs_ard("gp_ard.dat");
        for (int i = 0; i < 100; ++i)
        {
            Eigen::VectorXd v(2);
            v[0] = (i / 100.0) * 5.0 - 2.5;
            v[1] = 0.0;
            Eigen::VectorXd mu;
            double sigma;
            std::tie(mu, sigma) = gp_ard.query(v);
            ofs_ard << v.transpose() << " " << mu[0] << " " << sqrt(sigma) << std::endl;
        }
    }
    if (dim_in == 2 && dim_out == 2)
    {
        // write the predicted data in a file (e.g. to be plotted)
        std::ofstream ofs_ard("gp_ard.dat");
        for (int i = 0; i < 100; ++i)
        {
            Eigen::VectorXd v(2);
            v[0] = (i / 100.0) * 5.0 - 2.5;
            v[1] = 0.0;
            Eigen::VectorXd mu;
            double sigma;
            std::tie(mu, sigma) = gp_ard.query(v);
            ofs_ard << v[0] << " " << mu.transpose() << " " << sqrt(sigma) << std::endl;
        }
    }

    Eigen::VectorXd temp(2);
    temp[0] = 2;
    temp[1] = 1;
    Eigen::VectorXd temp_mu;
    double temp_sigma;
    std::tie(temp_mu, temp_sigma) = gp_ard.query(temp);

    std::cout << temp_mu.transpose() << " = mu for " << temp.transpose() << "\n";

    if (dim_in == 2 && dim_out == 2)
    {
        // write the predicted data in a file (e.g. to be plotted)
        std::ofstream ofs_surface("gp_surface_ard.dat");
        for (int i = 0; i < 20; ++i)
        {
            for (int j = 0; j < 20; ++j)
            {
                Eigen::VectorXd v(2);
                v[0] = (i / 20.0) * 5.0 - 2.5;
                v[1] = (j / 20.0) * 5.0 - 2.5;
                Eigen::VectorXd mu;
                double sigma;
                std::tie(mu, sigma) = gp_ard.query(v);
                // an alternative (slower) is to query mu and sigma separately:
                //  double mu = gp.mu(v)[0]; // mu() returns a 1-D vector
                //  double s2 = gp.sigma(v);
                ofs_surface << v.transpose() << " " << mu.transpose() << " " << sqrt(sigma) << std::endl;
            }
        }
    }

    std::cout << "noise = " << gp_ard.kernel_function().noise() << "\n";
    std::cout << "params = " << gp_ard.kernel_function().params().transpose() << "\n";
    std::cout << "mean = " << gp_ard.mean_function().h_params() << "\n";

    // write the data to a file (useful for plotting)
    std::ofstream ofs_data("data.dat");
    for (size_t i = 0; i < samples.size(); ++i)
        ofs_data << samples[i].transpose() << " " << observations[i].transpose() << std::endl;

    // Sometimes is useful to save an optimized GP
    gp_ard.save<serialize::TextArchive>("myGP");
    gp.save<serialize::TextArchive>("myGP_test");

    // Later we can load -- we need to make sure that the type is identical to the one saved
    gp_ard.load<serialize::TextArchive>("myGP");
    return 0;
}
