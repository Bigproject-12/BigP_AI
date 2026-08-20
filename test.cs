using System;

public class Test
{
    private const string PASSWORD = "1234";

    public static string GetPassword()
    {
        return PASSWORD;
    }

    public static string HashIt(string p)
    {
        return p.GetHashCode().ToString();
    }

    public static void Main()
    {
        Console.WriteLine("Password: " + PASSWORD);
        Console.WriteLine(HashIt(PASSWORD));
        Console.WriteLine(GetPassword());
    }
}