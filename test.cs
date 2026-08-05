using System;
using System.Data.SqlClient;

public class TestCode
{
    private const string API_SECRET = "hardcoded_secret_123";
    static readonly string DB_PASSWORD = Environment.GetEnvironmentVariable("DB_PASSWORD") ?? "1234";

    public void Login(string username, string password)
    {
        Console.WriteLine("Password entered: " + password);

        string query = "SELECT * FROM users WHERE username='" + username + "'";
        SqlCommand cmd = new SqlCommand(query, null);
    }
}