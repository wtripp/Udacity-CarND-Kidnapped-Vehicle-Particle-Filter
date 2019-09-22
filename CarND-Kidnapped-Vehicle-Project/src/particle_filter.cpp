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

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  num_particles = 100;  // TODO: Set the number of particles
  
  std::cout << "Initializing particles...\n" << std::endl;
  
  std::default_random_engine gen;
  double std_x = std[0];
  double std_y = std[1];
  double std_theta = std[2];
  
  std::normal_distribution<double> dist_x(x, std_x);
  std::normal_distribution<double> dist_y(y, std_y);
  std::normal_distribution<double> dist_theta(theta, std_theta);
  
  for(int i = 0; i < num_particles; i++) {
    Particle p;
    p.id = i;
    p.x = dist_x(gen);
    p.y = dist_y(gen);
    p.theta = dist_theta(gen);
    p.weight = 1.0;
    particles.push_back(p);
    weights.push_back(p.weight);
    //std::cout << "Initializing particle " << p.id+1 << "\n" <<
    //             "x: " << p.x << "\n"
    //             "y: " << p.y << "\n"
    //             "theta: " << p.theta << "\n"
    //             "weight: " << p.weight << "\n\n" << std::endl;
  }
  is_initialized = true;
  std::cout << "Particles initialized!\n\n" << std::endl;
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
  
  double std_pos_x = std_pos[0];
  double std_pos_y = std_pos[1];
  double std_pos_theta = std_pos[2];
  
  std::normal_distribution<double> dist_x(0, std_pos_x);
  std::normal_distribution<double> dist_y(0, std_pos_y);
  std::normal_distribution<double> dist_theta(0, std_pos_theta);
  
  // Add measurement predictions
  for(auto &p : particles) {
    std::cout << "Adding measurement predictions for particles...\n" << std::endl;
    //std::cout << "Adding measurement predictions for particle " << p.id+1 << "\n" << std::endl;
    
    //std::cout << "Initial x:" << p.x << "\n" <<
    //             "Initial y:" << p.y << "\n" <<
    //             "Initial theta:" << p.theta << "\n" << std::endl;

    // straightaways
    if(fabs(yaw_rate) < 0.001) {
      p.x += velocity * delta_t * cos(p.theta);
      p.y += velocity * delta_t * sin(p.theta);
    }
    
    // turns (nonzero yaw)
    else {
      p.x += velocity/yaw_rate * (sin(p.theta + yaw_rate*delta_t) - sin(p.theta));
      p.y += velocity/yaw_rate * (cos(p.theta) - cos(p.theta + yaw_rate*delta_t));
      p.theta += yaw_rate*delta_t;
    }

    //std::cout << "Predicted x:" << p.x << "\n" <<
    //             "Predicted y:" << p.y << "\n" <<
    //             "Predicted theta:" << p.theta << "\n" << std::endl;

    // Add Gaussian noise
    p.x += dist_x(gen);
    p.y += dist_y(gen);
    p.theta += dist_theta(gen);

    //std::cout << "Predicted x with noise:" << p.x << "\n" <<
    //             "Predicted y with noise:" << p.y << "\n" <<
    //             "Predicted theta with noise:" << p.theta << "\n" << std::endl;
  }
  std::cout << "Measurement predictions added!\n" << std::endl;
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
  std::cout << "Associating observations to closest landmarks...\n" << std::endl;
  double curr_dist, min_dist;
  int map_landmark_id;
  
  // For each observed landmark, find the closest predicted landmark.
  for(auto &obs : observations) {
    min_dist = 1E99;
    map_landmark_id = -1;
    for(auto &pred : predicted) {
      curr_dist = dist(obs.x, obs.y, pred.x, pred.y);
      if(curr_dist < min_dist) {
        min_dist = curr_dist;
        map_landmark_id = pred.id;
      }
    }
    // Assign ID of closest predicted landmark to the observed landmark.
    obs.id = map_landmark_id;
  }
  std::cout << "Associations complete!\n" << std::endl;
}

void ParticleFilter::setAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which to assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  std::cout << "Setting associations for visualization...\n" << std::endl;
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
  std::cout << "Associations set!\n" << std::endl;

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  std::cout << "Updating weights...\n" << std::endl;
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
  for(auto &p : particles) {

    // Transform observations to map coordinates
    vector<LandmarkObs> observations_map;
    for(auto &obs : observations) {
      double map_x = p.x + cos(p.theta)*obs.x - sin(p.theta)*obs.y;
      double map_y = p.y + sin(p.theta)*obs.x + cos(p.theta)*obs.y;
      LandmarkObs obs_map;
      obs_map.id = obs.id;
      obs_map.x = map_x;
      obs_map.y = map_y;
      observations_map.push_back(obs_map);
    }
    
    // Find landmarks in sensor range    
    vector<LandmarkObs> predicted;
    for(unsigned int i = 0; i < map_landmarks.landmark_list.size(); i++) {
      Map::single_landmark_s lm = map_landmarks.landmark_list[i];
      double distance = dist(p.x, p.y, lm.x_f, lm.y_f);
      if(distance <= sensor_range) {
        LandmarkObs pred = {lm.id_i, lm.x_f, lm.y_f};
        predicted.push_back(pred);
      }
    }
    
    // Associate observations with landmarks
    dataAssociation(predicted, observations_map);
    
    // Set assocations for visualization
    vector<int> associations;
    vector<double> sense_x;
    vector<double> sense_y;
    for(auto &obs_map : observations_map) {
      associations.push_back(obs_map.id);
      sense_x.push_back(obs_map.x);
      sense_y.push_back(obs_map.y);
    }
    setAssociations(p, associations, sense_x, sense_y);

    // Compute weights using multivariate Gaussian
    double std_x = std_landmark[0];
    double std_y = std_landmark[1];
    for(auto &obs_map : observations_map) {
      double x_part = pow(obs_map.x - p.x, 2) / (std_x * std_x);
      double y_part = pow(obs_map.y - p.y, 2) / (std_y * std_y);
      double w = 1.0 / (2 * M_PI * std_x * std_y) * exp(-0.5 * (x_part + y_part));
      //double x_part = pow(obs_map.x - p.x, 2) / (2 * std_x * std_x);
      //double y_part = pow(obs_map.y - p.y, 2) / (2 * std_y * std_y);
      //double w = 1 / (2*M_PI*std_x*std_y) * exp(-x_part - y_part);
      
      if(w > 0) {
        p.weight *= w;
      }
    }
    weights[p.id] = p.weight;
  }
  std::cout << "Weights updated!\n" << std::endl;
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  std::cout << "Resampling particles...\n" << std::endl;
  vector<Particle> resampled_particles(num_particles);
  std::default_random_engine gen;
  std::discrete_distribution<int> dist(weights.begin(), weights.end());
  
  for(int i = 0; i < num_particles; i++) {
    resampled_particles[i] = particles[dist(gen)];
  }
  particles = move(resampled_particles);
  std::cout << "Particles resampled!\n" << std::endl;
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