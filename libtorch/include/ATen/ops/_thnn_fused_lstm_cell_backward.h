#pragma once

// @generated by torchgen/gen.py from Function.h

#include <ATen/Context.h>
#include <ATen/DeviceGuard.h>
#include <ATen/TensorUtils.h>
#include <ATen/TracerMode.h>
#include <ATen/core/Generator.h>
#include <ATen/core/Reduction.h>
#include <ATen/core/Tensor.h>
#include <c10/core/Scalar.h>
#include <c10/core/Storage.h>
#include <c10/core/TensorOptions.h>
#include <c10/util/Deprecated.h>
#include <optional>
#include <string_view>



#include <ATen/ops/_thnn_fused_lstm_cell_backward_ops.h>

namespace at {


// aten::_thnn_fused_lstm_cell_backward(Tensor? grad_hy, Tensor? grad_cy, Tensor cx, Tensor cy, Tensor workspace, bool has_bias) -> (Tensor, Tensor, Tensor, Tensor, Tensor)
inline ::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor> _thnn_fused_lstm_cell_backward(const ::std::optional<at::Tensor> & grad_hy, const ::std::optional<at::Tensor> & grad_cy, const at::Tensor & cx, const at::Tensor & cy, const at::Tensor & workspace, bool has_bias) {
    return at::_ops::_thnn_fused_lstm_cell_backward::call(grad_hy, grad_cy, cx, cy, workspace, has_bias);
}

}
