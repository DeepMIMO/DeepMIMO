/* Column-major 4x4 matrix helpers, split out so the camera maths can be
 * tested without a browser: a wrong projection matrix shows up as a blank
 * canvas, which is hard to debug by looking at it.
 */

export const M4 = {
  mul(a, b) {
    const o = new Float32Array(16);
    for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) {
      let s = 0;
      for (let k = 0; k < 4; k++) s += a[k * 4 + j] * b[i * 4 + k];
      o[i * 4 + j] = s;
    }
    return o;
  },
  perspective(fovy, aspect, near, far) {
    const f = 1 / Math.tan(fovy / 2), o = new Float32Array(16);
    o[0] = f / aspect; o[5] = f; o[10] = (far + near) / (near - far);
    o[11] = -1; o[14] = (2 * far * near) / (near - far);
    return o;
  },
  ortho(l, r, b, t, near, far) {
    const o = new Float32Array(16);
    o[0] = 2 / (r - l); o[5] = 2 / (t - b); o[10] = -2 / (far - near);
    o[12] = -(r + l) / (r - l); o[13] = -(t + b) / (t - b);
    o[14] = -(far + near) / (far - near); o[15] = 1;
    return o;
  },
  // Gauss-Jordan rather than the cofactor expansion: shorter, and the matrices
  // it is asked to invert here are well conditioned.
  invert(m) {
    const a = Array.from(m), inv = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
    const at = (row, col) => a[col * 4 + row], set = (arr, row, col, v) => { arr[col * 4 + row] = v; };
    for (let i = 0; i < 4; i++) {
      let pivot = i;
      for (let r = i + 1; r < 4; r++) if (Math.abs(at(r, i)) > Math.abs(at(pivot, i))) pivot = r;
      if (Math.abs(at(pivot, i)) < 1e-12) return null;
      if (pivot !== i) {
        for (let c = 0; c < 4; c++) {
          let t = at(i, c); set(a, i, c, at(pivot, c)); set(a, pivot, c, t);
          t = inv[c * 4 + i]; inv[c * 4 + i] = inv[c * 4 + pivot]; inv[c * 4 + pivot] = t;
        }
      }
      const d = at(i, i);
      for (let c = 0; c < 4; c++) { set(a, i, c, at(i, c) / d); inv[c * 4 + i] /= d; }
      for (let r = 0; r < 4; r++) {
        if (r === i) continue;
        const f = at(r, i);
        if (!f) continue;
        for (let c = 0; c < 4; c++) {
          set(a, r, c, at(r, c) - f * at(i, c));
          inv[c * 4 + r] -= f * inv[c * 4 + i];
        }
      }
    }
    return new Float32Array(inv);
  },
  lookAt(eye, target, up) {
    const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    const norm = v => { const l = Math.hypot(...v) || 1; return [v[0] / l, v[1] / l, v[2] / l]; };
    const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
    const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    const z = norm(sub(eye, target)), x = norm(cross(up, z)), y = cross(z, x);
    const o = new Float32Array(16);
    o[0] = x[0]; o[4] = x[1]; o[8] = x[2]; o[12] = -dot(x, eye);
    o[1] = y[0]; o[5] = y[1]; o[9] = y[2]; o[13] = -dot(y, eye);
    o[2] = z[0]; o[6] = z[1]; o[10] = z[2]; o[14] = -dot(z, eye);
    o[15] = 1;
    return o;
  },
};
