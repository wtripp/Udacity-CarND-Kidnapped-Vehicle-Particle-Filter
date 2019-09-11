/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;
using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  unsigned int num_particles = 100;  // Set the number of particles
  
  // Gaussian distribution
  std::normal_distribution<double> dist_x(x, std[0]);
  std::normal_distribution<double> dist_y(y, std[1]);
  std::normal_distribution<double> dist_theta(theta, std[2]);
  
  std::default_random_engine gen;
  for(unsigned int i = 0; i < num_particles; i++) {
    Particle p;
    p.x = dist_x(gen);
    p.y = dist_y(gen);
    p.theta = dist_theta(gen);
    p.weight = 1.0;
    particles.push_back(p);
    weights.push_back(1.0);
  }

  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  std::default_random_engine gen;
  std::normal_distribution<double> dist_x(0, std_pos[0]);
  std::normal_distribution<double> dist_y(0, std_pos[1]);
  std::normal_distribution<double> dist_theta(0, std_pos[2]);
  
  for(unsigned int i = 0; i < num_particles; i++) {
    
    if (fabs(yaw_rate) > 0.00001) {
      // Add measurements to particles (for nozero yaw rate, i.e., turns)
      particles[i].x += velocity/yaw_rate * (sin(particles[i].theta + yaw_rate*delta_t) - sin(particles[i].theta));
      particles[i].y += velocity/yaw_rate * (cos(particles[i].theta) - cos(yaw_rate*delta_t - particles[i].theta));
      //particles[i].x += velocity * (sin(particles[i].theta) + yaw_rate*delta_t - sin(particles[i].theta)) / yaw_rate;
      //particles[i].y += velocity * (-cos(particles[i].theta) + yaw_rate*delta_t - cos(particles[i].theta)) / yaw_rate;
      particles[i].theta += yaw_rate * delta_t;
      
    } else {
      // Add measurements to particles (for near-zero yaw rate, i.e., straightaways)
      particles[i].x += velocity * delta_t * cos(particles[i].theta);
      particles[i].y += velocity * delta_t * sin(particles[i].theta);      
    }

    // Add noise
    particles[i].x += dist_x(gen);
    particles[i].y += dist_y(gen);
    particles[i].theta += dist_theta(gen);
    
  }

}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
  double MAX_DOUBLE = 1000000;
  double minDistance = MAX_DOUBLE;
  for(unsigned int i = 0; i < observations.size(); i++) {
    LandmarkObs obs = observations[i];
    for(unsigned int j = 0; j < predicted.size(); j++) {
      LandmarkObs pred = predicted[j];
      double d = dist(obs.x, obs.y, pred.x, pred.y);
      if (d < minDistance) {
        minDistance = d;
        observations[i].id = pred.id;
      }
    minDistance = MAX_DOUBLE;
    }
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
  
  vector<LandmarkObs> observationsInMapCoords;
  for(unsigned int particle = 0; particle < particles.size(); particle++) {

    
    for(unsigned int obs = 0; obs < observations.size(); obs++) {
      
      // Transform observations to map coordinates
      LandmarkObs obsInMapCoords;
      obsInMapCoords.id = observations[obs].id;
      obsInMapCoords.x = observations[obs].x * cos(particles[particle].theta) - observations[obs].y * sin(particles[particle].theta) + particles[particle].x;
      obsInMapCoords.y = observations[obs].x * sin(particles[particle].theta) - observations[obs].y * cos(particles[particle].theta) + particles[particle].y;
      observationsInMapCoords.push_back(obsInMapCoords);
    }
      
  // Obtain landmarks within sensor range
  vector<LandmarkObs> predictions;
  for(unsigned int i = 0; i < map_landmarks.landmark_list.size(); i++) {
    Map::single_landmark_s landmark = map_landmarks.landmark_list[i];
    double distance = dist(particles[particle].x, particles[particle].y, landmark.x_f, landmark.y_f);
    if(distance < sensor_range) {
      predictions.push_back(LandmarkObs{landmark.id_i, landmark.x_f, landmark.y_f});
    }
  }
    
  // Associate landmark predictions with observations
  dataAssociation(predictions, observationsInMapCoords);
  
  // Compute particle weight using multivariate Gaussian
  double weight = 1.0;
  for(unsigned int i = 0; i < observationsInMapCoords.size(); i++) {
    LandmarkObs obs = observationsInMapCoords[i];
    for(unsigned int j = 0; j < predictions.size(); j++) {
      LandmarkObs pred = predictions[j];
      if(obs.id == pred.id) {
        double dx_sq = pow(obs.x - pred.x, 2);
        double dy_sq = pow(obs.y - pred.y, 2);
        double sig_x = std_landmark[0] * std_landmark[0];
        double sig_y = std_landmark[1] * std_landmark[1];

        double gauss_norm = 1 / (2 * M_PI * std_landmark[0] * std_landmark[1]);
        double exponent = dx_sq/(2*sig_x) + dy_sq/(2*sig_y);

        weight = gauss_norm * exp(-exponent);
        particles[particle].weight *= weight;        
      }
    }
  weights[particle] = weight;
  }
}
}
void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  vector<Particle> resampled_particles(num_particles);
  random_device rd;
  default_random_engine gen(rd());
  discrete_distribution<int> weight_dist(weights.begin(), weights.end());

  for (unsigned int i = 0; i < num_particles; i++) {
    resampled_particles[i] = particles[weight_dist(gen)];
  }

  particles = move(resampled_particles);
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}