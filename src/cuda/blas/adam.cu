#ifndef __BLAS_ADAM_ZIG__
#define __BLAS_ADAM_ZIG__

#include "blas_utils.cu"

template <class T>
void __adam(
  StreamWrapper w,
  void* param,
  const void* grad,
  void* m,
  void* v,
  len_t n,
  T beta1,
  T beta2,
  T one_minus_beta1,
  T one_minus_beta2,
  T step_size,
  T epsilon
) {
  const auto _stream = __cast_stream(w);
  auto param_iter = static_cast<T*>(param);
  const auto grad_iter = static_cast<const T*>(grad);
  auto m_iter = static_cast<T*>(m);
  auto v_iter = static_cast<T*>(v);
  const auto counter = thrust::make_counting_iterator<len_t>(0ul);

  thrust::for_each(
    thrust::cuda::par.on(_stream),
    counter,
    counter + n,
    [=] __device__ (len_t i) {
      const T g = grad_iter[i];
      const T m_next = beta1 * m_iter[i] + one_minus_beta1 * g;
      const T v_next = beta2 * v_iter[i] + one_minus_beta2 * g * g;
      m_iter[i] = m_next;
      v_iter[i] = v_next;
      param_iter[i] -= step_size * m_next / (std::sqrt(v_next) + epsilon);
    }
  );
}

extern "C" void adam(
  dtype id,
  StreamWrapper w,
  void* param,
  const void* grad,
  void* m,
  void* v,
  len_t n,
  double beta1,
  double beta2,
  double one_minus_beta1,
  double one_minus_beta2,
  double step_size,
  double epsilon
) {
  switch (id) {
    case SINGLE:
      return __adam<f32>(
        w,
        param,
        grad,
        m,
        v,
        n,
        static_cast<f32>(beta1),
        static_cast<f32>(beta2),
        static_cast<f32>(one_minus_beta1),
        static_cast<f32>(one_minus_beta2),
        static_cast<f32>(step_size),
        static_cast<f32>(epsilon)
      );
    case DOUBLE:
      return __adam<f64>(
        w,
        param,
        grad,
        m,
        v,
        n,
        static_cast<f64>(beta1),
        static_cast<f64>(beta2),
        static_cast<f64>(one_minus_beta1),
        static_cast<f64>(one_minus_beta2),
        static_cast<f64>(step_size),
        static_cast<f64>(epsilon)
      );
  }
  CUDA_ASSERT(cudaPeekAtLastError());
}

#endif
