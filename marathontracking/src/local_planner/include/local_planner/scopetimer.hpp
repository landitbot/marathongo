#ifndef SCOPE_TIMER_HPP
#define SCOPE_TIMER_HPP

#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <thread>
#include <chrono>

class ScopedTimerMS
{
public:
    explicit ScopedTimerMS(std::string name)
        : name_(std::move(name)),
          start_(std::chrono::steady_clock::now()) {}

    ~ScopedTimerMS()
    {
        auto dur = std::chrono::steady_clock::now() - start_;
        double ms = std::chrono::duration<double, std::milli>(dur).count();
        std::cout << "[ScopedTimer] " << name_ << ": " << ms << " ms\n";
    }

private:
    std::string name_;
    std::chrono::steady_clock::time_point start_;
};

class ManualTimerMS
{
public:
    explicit ManualTimerMS(std::string name)
        : name_(std::move(name))
    {
    }

    ~ManualTimerMS()
    {
    }

    void start()
    {
        start_ = std::chrono::steady_clock::now();
    }

    double stop(bool print_out = true)
    {
        auto dur = std::chrono::steady_clock::now() - start_;
        double ms = std::chrono::duration<double, std::milli>(dur).count();
        if (print_out)
        {
            std::cout << "[ManualTimer] " << name_ << ": " << ms << " ms\n";
        }
        return ms;
    }

private:
    std::string name_;
    std::chrono::steady_clock::time_point start_;
};

#endif