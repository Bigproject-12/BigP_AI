#include <iostream>
#include <string>

std::string password = "1234";

std::string getPassword() {
    return password;
}

std::string hashIt(std::string p) {
    return std::to_string(std::hash<std::string>{}(p));
}

int main() {
    std::cout << "Password: " << password << std::endl;
    std::cout << hashIt(password) << std::endl;
    std::cout << getPassword() << std::endl;
    return 0;
}