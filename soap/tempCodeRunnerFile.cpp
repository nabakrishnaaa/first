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