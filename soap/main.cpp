#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
using namespace std;

struct Vec2 { double x, y; };

// The Rosenbrock "Banana" Loss
double get_loss(Vec2 p) {
    return std::pow(1.0 - p.x, 2) + 100.0 * std::pow(p.y - p.x * p.x, 2);
}

// Gradient of Rosenbrock
Vec2 get_grad(Vec2 p) {
    double dx = -2.0 * (1.0 - p.x) - 400.0 * p.x * (p.y - p.x * p.x);
    double dy = 200.0 * (p.y - p.x * p.x);
    return {dx, dy};
}

class AdamW {
public:
    double lr = 0.02, b1 = 0.9, b2 = 0.999;
    Vec2 m = {0,0}, v = {0,0};
    int t = 0;
    void step(Vec2& p, Vec2 g) {
        t++;
        m.x = b1 * m.x + (1-b1) * g.x; m.y = b1 * m.y + (1-b1) * g.y;
        v.x = b2 * v.x + (1-b2) * g.x * g.x; v.y = b2 * v.y + (1-b2) * g.y * g.y;
        double m_h_x = m.x / (1 - std::pow(b1, t)), m_h_y = m.y / (1 - std::pow(b1, t));
        double v_h_x = v.x / (1 - std::pow(b2, t)), v_h_y = v.y / (1 - std::pow(b2, t));
        p.x -= lr * m_h_x / (std::sqrt(v_h_x) + 1e-8);
        p.y -= lr * m_h_y / (std::sqrt(v_h_y) + 1e-8);
    }
};

class SOAP {
public:
    double lr = 0.02;
    Vec2 m = {0,0}, v = {0,0};
    double Q[2][2] = {{1,0},{0,1}}; // Eigenbasis
    int t = 0;
    void step(Vec2& p, Vec2 g) {
        t++;
        // Rotation into Eigenbasis
        double g_rot_x = Q[0][0] * g.x + Q[1][0] * g.y;
        double g_rot_y = Q[0][1] * g.x + Q[1][1] * g.y;
        m.x = 0.9 * m.x + 0.1 * g_rot_x; m.y = 0.9 * m.y + 0.1 * g_rot_y;
        v.x = 0.99 * v.x + 0.01 * g_rot_x * g_rot_x; v.y = 0.99 * v.y + 0.01 * g_rot_y * g_rot_y;
        double u_rot_x = (m.x / (1 - std::pow(0.9, t))) / (std::sqrt(v.x / (1 - std::pow(0.99, t))) + 1e-8);
        double u_rot_y = (m.y / (1 - std::pow(0.9, t))) / (std::sqrt(v.y / (1 - std::pow(0.99, t))) + 1e-8);
        // Rotation back to weight space
        p.x -= lr * (Q[0][0] * u_rot_x + Q[0][1] * u_rot_y);
        p.y -= lr * (Q[1][0] * u_rot_x + Q[1][1] * u_rot_y);
        if (t % 5 == 0) { // Update Eigen-rotation
            double a = 0.02; double c = std::cos(a), s = std::sin(a);
            double t00 = Q[0][0]*c - Q[0][1]*s; double t01 = Q[0][0]*s + Q[0][1]*c;
            double t10 = Q[1][0]*c - Q[1][1]*s; double t11 = Q[1][0]*s + Q[1][1]*c;
            Q[0][0] = t00; Q[0][1] = t01; Q[1][0] = t10; Q[1][1] = t11;
        }
    }
};

int main() {
    Vec2 p_adam = {-1.5, 2.0}, p_soap = {-1.5, 2.0};
    AdamW opt_a; SOAP opt_s;
    std::ofstream out("result.csv");
    out << "step,ax,ay,aloss,sx,sy,sloss\n";
    for (int i = 0; i < 250; ++i) {
        out << i << "," << p_adam.x << "," << p_adam.y << "," << get_loss(p_adam) << ","
            << p_soap.x << "," << p_soap.y << "," << get_loss(p_soap) << "\n";
        opt_a.step(p_adam, get_grad(p_adam));
        opt_s.step(p_soap, get_grad(p_soap));
    }
    out.close();
    std::cout << "Done! Results saved in result.csv\n";
    return 0;
}