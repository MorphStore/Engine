/**
 * author: Patrick Damme
 */

#ifndef STOPWATCH_H
#define STOPWATCH_H

#include <chrono>
#include <ctime>

    /**
     * The superclass of all stop watches. A stop watch is intended for
     * measuring time (e.g. run times of transformation algorithms). After its
     * creation, a stop watch sw should be used the following way:
     *
     * sw.start();
     * // do something
     * sw.stop();
     * double runtimeInSec = sw.duration();
     *
     * The same stop watch instance can be reused by doing:
     *
     * sw.reset();
     *
     * afterwards.
     */
template<class T>
class StopWatch {
private:

    enum state_t {
        init, started, stopped
    } state;

protected:
    T startTime, endTime;
    virtual T now() const = 0;
    virtual double diff() const = 0;

public:

    StopWatch() : state(init) {
        //
    }

    void start() {
        if (state != init)
            throw "illegal stop watch state";

        state = started;
        startTime = now();
    }

    void stop() {
        if (state != started)
            throw "illegal stop watch state";

        state = stopped;
        endTime = now();
    }

    double duration() const {
        if (state != stopped)
            throw "illegal stop watch state";

        return diff();
    }

    void reset() {
        state = init;
    }
};

/**
 * This stop watch is based on the std::chrono::high_resolution_clock from the 
 * C++11 <chrono> header.
 */
class WallClockStopWatch : public StopWatch<std::chrono::high_resolution_clock::time_point> {
protected:

    std::chrono::high_resolution_clock::time_point now() const {
        return std::chrono::high_resolution_clock::now();
    }

    double diff() const {
        return static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count()) / 1000000;
    }
};

/**
 * This stop watch is based on the <ctime> header function clock().
 */
class CpuStopWatch : public StopWatch<clock_t> {
protected:

    clock_t now() const {
        return clock();
    }

    double diff() const {
        return static_cast<double>(endTime - startTime) / CLOCKS_PER_SEC;
    }
};

#endif  /* STOPWATCH_H */