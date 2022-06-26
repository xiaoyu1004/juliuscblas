#ifndef JULIUS_DEFINED_HPP
#define JULIUS_DEFINED_HPP

// export
#if defined(_MSC_VER)

#if defined(BUILDING_JULIUS_DLL)
#define JULIUS_PUBLIC __declspec(dllexport)
#elif defined(USING_JULIUS_DLL)
#define JULIUS_PUBLIC __declspec(dllimport)
#else
#define JULIUS_PUBLIC
#endif // BUILDING_JULIUS_DLL

#else // linux
#define JULIUS_PUBLIC __attribute__((visibility("default")))
#endif

// log
#define DEFAULT_TAG "julius_cblas"

#ifdef DEBUG
#define LOGDT(fmt, tag, ...) fprintf(stdout, ("D/%s: %s [File %s][Line %d] " fmt), tag, __FUNCTION__, __FILE__, __LINE__, ##__VA_ARGS__)
#endif
#define LOGIT(fmt, tag, ...) fprintf(stdout, ("I/%s: %s [File %s][Line %d] " fmt), tag, __FUNCTION__, __FILE__, __LINE__, ##__VA_ARGS__)
#define LOGWT(fmt, tag, ...) fprintf(stdout, ("W/%s: %s [File %s][Line %d] " fmt), tag, __FUNCTION__, __FILE__, __LINE__, ##__VA_ARGS__)
#define LOGET(fmt, tag, ...) fprintf(stderr, ("E/%s: %s [File %s][Line %d] " fmt), tag, __FUNCTION__, __FILE__, __LINE__, ##__VA_ARGS__)

#ifdef DEBUG
#define LOGD(fmt, ...) LOGDT(fmt, DEFAULT_TAG, ##__VA_ARGS__)
#endif
#define LOGI(fmt, ...) LOGIT(fmt, DEFAULT_TAG, ##__VA_ARGS__)
#define LOGW(fmt, ...) LOGWT(fmt, DEFAULT_TAG, ##__VA_ARGS__)
#define LOGE(fmt, ...)                          \
    {                                           \
        LOGET(fmt, DEFAULT_TAG, ##__VA_ARGS__); \
        std::terminate();                       \
    }
#define LOGE_IF(cond, fmt, ...)   \
    if (cond)                     \
    {                             \
        LOGE(fmt, ##__VA_ARGS__); \
    }

// check
#define CHECK_BINARY_OP(name, op, a, b) \
    if (!((a)op(b)))                    \
    LOGW("CHECK %s FAILED(%s %s %s vs. %d %s %d)\n", #name, #a, #op, #b, (a), #op, (b))

#define CHECK_LT(x, y) CHECK_BINARY_OP(_LT, <, x, y)
#define CHECK_GT(x, y) CHECK_BINARY_OP(_GT, >, x, y)
#define CHECK_LE(x, y) CHECK_BINARY_OP(_LE, <=, x, y)
#define CHECK_GE(x, y) CHECK_BINARY_OP(_GE, >=, x, y)
#define CHECK_EQ(x, y) CHECK_BINARY_OP(_EQ, ==, x, y)
#define CHECK_NE(x, y) CHECK_BINARY_OP(_NE, !=, x, y)
#define CHECK_NOTNULL(x) ((x) == NULL ? LOGE("Check  notnull: %s\n", #x) : (x))

#define NOT_IMPLEMENTED LOGE("Not Implemented Yet.\n");

#endif // JULIUS_DEFINED_HPP