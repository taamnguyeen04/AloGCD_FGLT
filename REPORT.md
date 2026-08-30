# BÁO CÁO SO SÁNH 3 CHẾ ĐỘ (BASE / MODE 1 / MODE 2) — MÔ HÌNH BACON

**Datasets:** CIFAR-10, CUB-200
**Imbalance ratio (imb):** 100
**Epochs tối đa:** 200
**Pseudo update freq:** 10 epoch (19 lần update trong 200 epoch, tại epoch 10, 20, ..., 190)
**Pseudo threshold (mode 2):** 0.90
**Top ratio (mode 1):** 0.80 (giữ top 80% theo confidence)

---

## 0. Ý TƯỞNG HAI CHẾ ĐỘ PSEUDO-LABELING

### Mode 1 — "Đẩy thêm vào class mạnh nhất"

**Ý tưởng:** model tự tin nhất về class nào thì bồi thêm mẫu cho class đó.

Cách làm mỗi vòng cập nhật (10 epoch/lần):

1. **Đánh giá acc của từng known class** trên test set.
2. **Chọn class có acc cao nhất** (gọi là class "mạnh nhất").
3. **Quét tập unlabel**, giữ lại những ảnh mà model dự đoán đúng vào class mạnh nhất đó.
4. **Sắp xếp các ảnh đó theo độ tin cậy giảm dần, lấy top 80%** cao nhất.
5. **Gán nhãn class mạnh nhất cho các ảnh được chọn, thêm vào tập labeled** để train tiếp.

### Mode 2 — "Quét ngưỡng tin cậy cao"

**Ý tưởng:** lấy mọi ảnh unlabel mà model rất tự tin, không giới hạn ở 1 class.

Cách làm mỗi vòng cập nhật (10 epoch/lần):

1. **Quét tập unlabel**, tính độ tin cậy (max softmax) cho từng ảnh.
2. **Chỉ giữ ảnh có độ tin cậy ≥ 0.90** và thuộc known class.
3. **Với mỗi class, lấy tối đa 500 mẫu có độ tin cậy cao nhất** (giới hạn theo class).
4. **Gán nhãn tương ứng cho các ảnh được chọn, thêm vào tập labeled** để train tiếp.


### So sánh nhanh

| | Mode 1 | Mode 2 |
| --- | --- | --- |
| Chọn class | 1 class mạnh nhất | Mọi class thỏa điều kiện |
| Tiêu chí chọn ảnh | Đúng class đó + top 80% | max_confidence ≥ 0.90 |

---

## 1. TÓM TẮT KẾT QUẢ (test set, best checkpoint)

### 1.1 Old / New / All (3 chỉ số chính)

<table>
  <thead>
    <tr><th>Dataset</th><th>Mode</th><th>All</th><th>Old</th><th>New</th></tr>
  </thead>
  <tbody>
    <tr><td>CIFAR-10</td><td>base</td><td>91.1</td><td><b>94.8</b></td><td>88.2</td></tr>
    <tr><td>CIFAR-10</td><td><b>mode 1</b></td><td><b>91.3</b></td><td>93.7</td><td><b>89.0</b></td></tr>
    <tr><td>CIFAR-10</td><td>mode 2</td><td><b>91.3</b></td><td><u>94.3</u></td><td><u>88.3</u></td></tr>
    <tr><td>CIFAR-10</td><td>BaCon-O</td><td>80.7</td><td>83.3</td><td>78.0</td></tr>
    <tr><td>CIFAR-10</td><td>BaCon-S</td><td>91.1</td><td>94.2</td><td>88.1</td></tr>
    <tr><td colspan="5" style="height:14px;border:none;background:transparent"></td></tr>
    <tr><td>CUB-200</td><td>base</td><td><u>55.2</u></td><td><u>64.6</u></td><td><u>46.1</u></td></tr>
    <tr><td>CUB-200</td><td><b>mode 1</b></td><td><b>57.1</b></td><td><b>66.3</b></td><td><b>48.0</b></td></tr>
    <tr><td>CUB-200</td><td>mode 2</td><td>55.0</td><td>64.4</td><td>45.9</td></tr>
  </tbody>
</table>

### 1.2 Many / Med / Few (phân nhóm head/mid/tail)

<table>
  <thead>
    <tr><th>Dataset</th><th>Mode</th><th>K_Many</th><th>K_Med</th><th>K_Few</th><th>U_Many</th><th>U_Med</th><th>U_Few</th></tr>
  </thead>
  <tbody>
    <tr><td>CIFAR-10</td><td>base</td><td><b>98.2</b></td><td><u>91.9</u></td><td><b>96.1</b></td><td><b>95.8</b></td><td>80.3</td><td>90.2</td></tr>
    <tr><td>CIFAR-10</td><td><b>mode 1</b></td><td>94.7</td><td><b>92.6</b></td><td><u>94.3</u></td><td>87.8</td><td><b>89.5</b></td><td>89.2</td></tr>
    <tr><td>CIFAR-10</td><td>mode 2</td><td><u>95.6</u></td><td><u>92.6</u></td><td>95.5</td><td><u>95.6</u></td><td><u>80.6</u></td><td><b>92.4</b></td></tr>
    <tr><td colspan="8" style="height:14px;border:none;background:transparent"></td></tr>
    <tr><td>CUB-200</td><td>base</td><td>n/a</td><td>64.6</td><td>n/a</td><td>n/a</td><td>47.5</td><td>25.3</td></tr>
    <tr><td>CUB-200</td><td><b>mode 1</b></td><td>n/a</td><td><b>66.3</b></td><td>n/a</td><td>n/a</td><td><b>49.0</b></td><td><b>32.6</b></td></tr>
    <tr><td>CUB-200</td><td>mode 2</td><td>n/a</td><td>64.4</td><td>n/a</td><td>n/a</td><td>47.0</td><td>29.8</td></tr>
  </tbody>
</table>

### 1.3 Per-class acc (known classes)

Removed.

### 1.4 CUB-200 per-decile (0 = head, 9 = tail)

Removed.

---

## 2. PER-CLASS ACC THEO EPOCH (CIFAR-10, mỗi 10 epoch)

### 2.1 base

| epoch | C0    | C1    | C2    | C3    | C4    |
| ----- | ----- | ----- | ----- | ----- | ----- |
| 9     | 0.938 | <b>0.964</b> | 0.889 | <u>0.946</u> | 0.872 |
| 29    | <u>0.944</u> | <b>0.967</b> | 0.864 | 0.659 | 0.923 |
| 59    | <u>0.956</u> | 0.951 | 0.491 | 0.016 | <b>0.957</b> |
| 89    | <u>0.949</u> | <b>0.970</b> | 0.930 | 0.937 | 0.918 |
| 119   | <u>0.956</u> | <b>0.970</b> | 0.921 | 0.936 | 0.908 |
| 149   | 0.796 | <b>0.968</b> | 0.887 | 0.929 | <u>0.936</u> |
| 179   | 0.792 | <b>0.967</b> | 0.889 | <u>0.931</u> | 0.928 |
| 199   | 0.797 | <b>0.968</b> | 0.886 | <u>0.932</u> | 0.926 |

### 2.2 mode 1

| epoch | C0    | C1    | C2    | C3    | C4    | update             |
| ----- | ----- | ----- | ----- | ----- | ----- | ------------------ |
| 9     | 0.941 | <b>0.961</b> | 0.888 | <u>0.943</u> | 0.881 | iter 1: +248 → C1 |
| 19    | <u>0.959</u> | <b>0.965</b> | 0.912 | 0.952 | 0.900 | iter 2: +40 → C1  |
| 29    | <b>0.961</b> | 0.555 | 0.908 | <u>0.950</u> | 0.944 | iter 3: +51 → C1  |
| 39    | <u>0.963</u> | <b>0.964</b> | 0.497 | 0.943 | 0.940 | iter 4: +500 → C0 |
| 49    | <b>0.978</b> | 0.937 | 0.935 | 0.918 | <u>0.947</u> | iter 5: +23 → C1  |
| 59    | <u>0.956</u> | <b>0.962</b> | 0.936 | 0.927 | 0.923 | iter 6: +6 → C1   |
| 69    | <b>0.964</b> | <u>0.962</u> | 0.923 | 0.929 | 0.922 | iter 7: +5 → C1   |
| 79    | <u>0.950</u> | <b>0.980</b> | 0.918 | 0.903 | 0.926 | (no update)        |
| 89    | <u>0.951</u> | <b>0.962</b> | 0.897 | 0.605 | 0.944 | (no update)        |
| 109   | <u>0.957</u> | <b>0.967</b> | 0.518 | 0.892 | 0.939 | (no update)        |
| 119   | 0.925 | <b>0.990</b> | 0.532 | 0.905 | 0.921 | (no update)        |
| 129   | 0.916 | <b>0.966</b> | 0.465 | 0.062 | <u>0.928</u> | (no update)        |
| 149   | 0.913 | <b>0.989</b> | <u>0.922</u> | 0.857 | 0.907 | (no update)        |
| 169   | 0.913 | <b>0.966</b> | 0.881 | 0.588 | <u>0.935</u> | iter 17: +1 → C1  |
| 189   | 0.913 | <b>0.985</b> | 0.878 | 0.555 | 0.913 | —                 |
| 199   | 0.911 | <b>0.985</b> | 0.877 | 0.555 | 0.913 | —                 |

### 2.3 mode 2

| epoch | C0    | C1    | C2    | C3    | C4    |
| ----- | ----- | ----- | ----- | ----- | ----- |
| 9     | 0.594 | <b>0.964</b> | 0.866 | 0.000 | <u>0.897</u> |
| 29    | <b>0.958</b> | <u>0.954</u> | 0.920 | 0.000 | 0.923 |
| 59    | <u>0.952</u> | <b>0.965</b> | 0.912 | 0.946 | 0.921 |
| 89    | <u>0.953</u> | <b>0.994</b> | 0.474 | 0.940 | 0.911 |
| 119   | 0.925 | <b>0.964</b> | <u>0.939</u> | 0.010 | 0.907 |
| 149   | <u>0.945</u> | <b>0.963</b> | 0.923 | 0.937 | 0.892 |
| 179   | 0.940 | <b>0.963</b> | 0.925 | <u>0.941</u> | 0.892 |
| 199   | <u>0.940</u> | <b>0.964</b> | 0.925 | 0.937 | 0.887 |

---

## 3. PSEUDO AUDIT — BEST CLASS MỖI ITER

### 3.1 CIFAR-10 — labeled set thật:

<table>
  <thead>
    <tr><th>Class</th><th>n_labeled</th><th>Vai trò</th><th>Số iter được chọn best</th><th>% iter</th></tr>
  </thead>
  <tbody>
    <tr><td>1</td><td>63</td><td>TAIL2 (minority)</td><td>174 / 200</td><td>87.0%</td></tr>
    <tr><td>0</td><td>2250</td><td>HEAD nhất</td><td>24 / 200</td><td>12.0%</td></tr>
    <tr><td>3</td><td>23</td><td>TAIL nhất</td><td>2 / 200</td><td>1.0%</td></tr>
  </tbody>
</table>

### 3.2 CUB-200 — labeled set 100 class, 15 mẫu/class

<table>
  <thead>
    <tr><th>Class</th><th>n_test</th><th>Vai trò (by idx)</th><th>Số iter chọn</th><th>% iter</th></tr>
  </thead>
  <tbody>
    <tr><td>3</td><td>30</td><td>head (idx 3)</td><td>97</td><td>48.5%</td></tr>
    <tr><td>18</td><td>29</td><td>head (idx 18)</td><td>30</td><td>15.0%</td></tr>
    <tr><td>11</td><td>26</td><td>head (idx 11)</td><td>22</td><td>11.0%</td></tr>
    <tr><td>13</td><td>30</td><td>head (idx 13)</td><td>14</td><td>7.0%</td></tr>
    <tr><td>52</td><td>30</td><td>mid (idx 52)</td><td>8</td><td>4.0%</td></tr>
    <tr><td>27</td><td>29</td><td>head (idx 27)</td><td>8</td><td>4.0%</td></tr>
    <tr><td>21</td><td>26</td><td>head (idx 21)</td><td>7</td><td>3.5%</td></tr>
    <tr><td>34</td><td>30</td><td>mid (idx 34)</td><td>4</td><td>2.0%</td></tr>
    <tr><td>73</td><td>30</td><td>tail (idx 73)</td><td>4</td><td>2.0%</td></tr>
    <tr><td>51</td><td>30</td><td>mid (idx 51)</td><td>2</td><td>1.0%</td></tr>
    <tr><td>13 class khác</td><td>—</td><td>—</td><td>1 mỗi cái</td><td>5.0%</td></tr>
  </tbody>
</table>

---

## 4. PSEUDO UPDATE BREAKDOWN — MODE 1

### 4.1 CIFAR-10 mode 1 (10/19 iter thực sự thêm pseudo, tổng 874 mẫu)

<table>
  <thead>
    <tr><th>iter</th><th>class</th><th>n_labeled (trước)</th><th>+pseudo</th><th>n_labeled (sau)</th><th>acc class (epoch trước)</th><th>verdict</th></tr>
  </thead>
  <tbody>
    <tr><td>1</td><td>1</td><td>63</td><td>248</td><td>311</td><td>0.961</td><td>HEAVILY BIASED</td></tr>
    <tr><td>2</td><td>1</td><td>311</td><td>40</td><td>351</td><td>0.965</td><td>HEAVILY BIASED</td></tr>
    <tr><td>3</td><td>1</td><td>351</td><td>51</td><td>402</td><td>0.555</td><td>HEAVILY BIASED</td></tr>
    <tr><td>4</td><td>0</td><td>2250</td><td>500</td><td>2750</td><td>0.963</td><td>OK (100% correct)</td></tr>
    <tr><td>5</td><td>1</td><td>402</td><td>23</td><td>425</td><td>0.937</td><td>HEAVILY BIASED</td></tr>
    <tr><td>6</td><td>1</td><td>425</td><td>6</td><td>431</td><td>0.962</td><td>HEAVILY BIASED</td></tr>
    <tr><td>7</td><td>1</td><td>431</td><td>5</td><td>436</td><td>0.962</td><td>HEAVILY BIASED</td></tr>
    <tr><td>8</td><td>1</td><td>436</td><td>0</td><td>436</td><td>0.980</td><td>(no update)</td></tr>
    <tr><td>10</td><td>1</td><td>436</td><td>0</td><td>436</td><td>0.966</td><td>(no update)</td></tr>
    <tr><td>17</td><td>1</td><td>436</td><td>1</td><td>437</td><td>0.966</td><td>HEAVILY BIASED</td></tr>
  </tbody>
</table>

### 4.2 CUB-200 mode 1 (11/19 iter thực sự thêm pseudo, tổng 338 mẫu, 5 iter "no update")

<table>
  <thead>
    <tr><th>iter</th><th>class</th><th>n_labeled (trước)</th><th>+pseudo</th><th>n_labeled (sau)</th><th>acc class (epoch trước)</th><th>verdict</th></tr>
  </thead>
  <tbody>
    <tr><td>1</td><td>73</td><td>15</td><td>2</td><td>17</td><td>1.000</td><td>OK</td></tr>
    <tr><td>2</td><td>3</td><td>15</td><td>23</td><td>38</td><td>1.000</td><td>HEAVILY BIASED</td></tr>
    <tr><td>3</td><td>11</td><td>15</td><td>11</td><td>26</td><td>0.962</td><td>OK</td></tr>
    <tr><td>4</td><td>51</td><td>15</td><td>10</td><td>25</td><td>0.000</td><td>HEAVILY BIASED</td></tr>
    <tr><td>5</td><td>21</td><td>15</td><td>34</td><td>49</td><td>0.308</td><td>HEAVILY BIASED</td></tr>
    <tr><td>6</td><td>27</td><td>15</td><td>248</td><td>263</td><td>0.966</td><td>HEAVILY BIASED</td></tr>
    <tr><td>7</td><td>18</td><td>15</td><td>10</td><td>25</td><td>0.966</td><td>OK</td></tr>
    <tr><td>8</td><td>11</td><td>26</td><td>0</td><td>26</td><td>0.692</td><td>(no update)</td></tr>
    <tr><td>10</td><td>27</td><td>263</td><td>0</td><td>263</td><td>0.138</td><td>(no update)</td></tr>
    <tr><td>13</td><td>18</td><td>25</td><td>0</td><td>25</td><td>0.966</td><td>(no update)</td></tr>
    <tr><td>16</td><td>34</td><td>15</td><td>0</td><td>15</td><td>0.933</td><td>(no update)</td></tr>
  </tbody>
</table>
