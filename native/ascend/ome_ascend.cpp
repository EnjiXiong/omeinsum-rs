#include "ome_ascend.h"

#include <acl/acl.h>
#include <aclnnop/aclnn_add.h>
#include <aclnnop/aclnn_matmul.h>
#include <aclnnop/aclnn_permute.h>
#include <aclnnop/aclnn_sub.h>

#include <string>

namespace {
thread_local std::string last_error;
}

extern "C" const char *ome_ascend_last_error(void) {
    return last_error.c_str();
}
