#pragma once
#include <iostream>
#include <vector>
#include <thread>
#include <functional>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <array>

template<typename T, size_t Capacity>
class LockFreeRingBuffer {
public:
    bool push(const T& item) {
        size_t current_tail = tail.load(std::memory_order_relaxed);
        size_t current_head = head.load(std::memory_order_acquire);

        if (current_tail - current_head >= Capacity) return false;

        size_t index = current_tail % Capacity;
        buffer[index] = item;
        tail.store(current_tail + 1, std::memory_order_release);
        return true;
    }

    bool pop(T& item) {
        size_t current_head = head.load(std::memory_order_acquire);
        size_t current_tail = tail.load(std::memory_order_acquire);

        if (current_head == current_tail) return false;

        size_t index = current_head % Capacity;
        item = buffer[index];
        head.store(current_head + 1, std::memory_order_release);
        return true;
    }

    bool empty() const {
        return head.load(std::memory_order_acquire) == tail.load(std::memory_order_acquire);
    }

private:
    std::atomic<size_t> head{ 0 }, tail{ 0 };
    std::array<T, Capacity> buffer;
};

class ThreadPool {
public:
    ThreadPool() {
        for (auto& t : workers)
            t = std::thread(&ThreadPool::workerThread, this);
    }

    ~ThreadPool() {
        {
            std::lock_guard<std::mutex> lock(mtx);
            done = true;
        }
        cv.notify_all();
        for (auto& t : workers)
            if (t.joinable())
                t.join();
    }

    template<typename F, typename... Args>
    bool enqueue(F&& f, Args&&... args) {
        auto task = [this, func = std::bind(std::forward<F>(f), std::forward<Args>(args)...)] {
            try {
                func();
            }
            catch (...) {
                std::cerr << "[ThreadPool] Unknown Exception in Task." << std::endl;
            }
            {
                std::lock_guard<std::mutex> lock(mtx);
                activeTasks--;
            }
            cv.notify_all();
            };

        {
            std::lock_guard<std::mutex> lock(mtx);
            activeTasks++;
        }

        if (!tasks.push(task)) {
            return false;
        }

        cv.notify_one();
        return true;
    }

    void waitForAllTasks() {
        std::unique_lock<std::mutex> lock(mtx);
        cv.wait(lock, [this] {
            return activeTasks == 0 && tasks.empty();
            });
    }

private:
    void workerThread() {
        while (true) {
            std::function<void()> task;
            {
                std::unique_lock<std::mutex> lock(mtx);
                cv.wait(lock, [this] { return done || !tasks.empty(); });
                if (done && tasks.empty())
                    return;
                if (!tasks.pop(task))
                    continue;
            }
            task();
        }
    }

    LockFreeRingBuffer<std::function<void()>, 256> tasks;
    std::array<std::thread, 8> workers;
    std::atomic<int> activeTasks{ 0 };
    std::mutex mtx;
    std::condition_variable cv;
    bool done=false;
};
