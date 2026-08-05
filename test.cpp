#include <iostream>
#include <cstdlib>
#include <cstdio>

const std::string API_SECRET = "hardcoded_secret_123";

void login(const char* username, const char* password) {
    std::cout << "Password entered: " << password << std::endl;

    std::string cmd = "ls -la " + std::string(username);
    system(cmd.c_str());
}
