#include <stdio.h>
#include <stdlib.h>
#include <string.h>

char *api_secret = "hardcoded_secret_123";

void login(char *username, char *password) {
    printf("Password entered: %s", password);

    char cmd[100];
    sprintf(cmd, "ls -la %s", username);
    system(cmd);
}
