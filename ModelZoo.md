#### TSN

| Model     | Dataset      | Frames | Input    | Top-1 | Top-1\* | mAP  | mAP\*  | Link                                                         |
| :-------- | :----------- | :----- | -------- | :---- | ------- | :--- | ---- | ------------------------------------------------------------ |
| resnet101 | MMit         | 5      | 224\*224 | -     | -       | 58.9 | 60.7 | [model](https://drive.google.com/open?id=1fM53qYCceZEpdtnc06XmXjyMrb7Is7_a) |
| resnet50  | Kinetics-600 | 8      | 224\*224 | 67.5  | 70.0    | -    | -    | [model](https://drive.google.com/open?id=1PWiCd15_VnBAwh3n-zzqzbGi6xVIhAeN) |




#### TIN

| Model     | Dataset     |      | Input    | Top-1 | Top-1\* | mAP  | mAP\*  | Link                                                         |
| :-------- | :---------- | ---- | :------- | :---- | :---- | :--- | ---- | ------------------------------------------------------------ |
| resnet50  | MMit        | 8    | 224\*224 | -     | -     | 62.2 | 62.8 | [model](https://drive.google.com/open?id=1f1kXH0cv7rJyc590ksasQ4GCVD9B-Jx8) |
| resnet50  | MMit        | 16   | 224\*224 | -     | -     | 62.5 | 62.9 | [model](https://drive.google.com/open?id=1Tqsfqqol5udoVX0KhnexGZzzGZs0G9MZ) |
| resnet101 | MMit        | 8    | 224\*224 | -     | -     | 62.2 | 63.0 | [model](https://drive.google.com/open?id=140dJeXaUVvqnLyI8h4wEYxxgLpXRBScF) |
| resnet50  | Something   | 8    | 224\*224 | 46.0  | 47.1  | -    | -    | [model](https://drive.google.com/open?id=1xibYXjyvOsteoJNmSylXD9E7MKfIQYJQ) |


#### SlowFast

| Model       | Dataset      | Frames      | Input    | Top-1 | Top-1\* | mAP  | mAP\*  | Link    |
| :---------- | :----------- | :---------- | -------- | :---- | :---- | :--- | ---- | ------- |
| SlowFast101 | Kinetics-700 | 64(8 \* 8)  | 112\*112 | -     | 65.2  | -    | -    | [model](https://drive.google.com/open?id=1IITbtSIAIfhHiZPtwB5GtSzq2evp20Ga) |
| SlowFast101 | MMit         | 64(8 \* 8) | 112\*112 | -     | -     | 59.9 | 61.5 | [model](https://drive.google.com/open?id=1dDilpoOGFpLql0a5M8gyEGkbX4JtpfRN) |
| SlowFast50  | Kinetics-600 | 64(8 \* 8) | 112\*112 | 70.0  | 77.5  | -    | -    | [model](https://drive.google.com/open?id=1QPh3tKH9VzuaHr0oG3va3yDdKeomqLkm) |
| SlowFast50  | Kinetics-600 | 64(8 \* 8)  | 224\*224 | 72.3 | 79.8 | - | - | [model](https://drive.google.com/open?id=1WnuJxNHv1E81rtP-GNviVhIOVQffvl2s) |



**\* :** Means using multi crops and multi clips (3 * 10)  when testing

**TSM** Models can refer to [Github](https://github.com/mit-han-lab/temporal-shift-module)

**MMit :** Multi-Moments in Time
