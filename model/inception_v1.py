import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import os
from utils.utils import Conv2d_KSE
# Data handling:
# normalize = transforms.Normalize(mean=[0.4588, 0.4588, 0.4588],
#                                  std=[1, 1, 1])
# ...
# val_loader = torch.utils.data.DataLoader(
#         datasets.ImageFolder(valdir, transforms.Compose([
#             transforms.Scale(256),
#             transforms.CenterCrop(224),
#             transforms.ToTensor(),
#             normalize
#         ])),
#         batch_size=args.batch_size, shuffle=False,
#         num_workers=args.workers, pin_memory=True)

def layer_init(m):
    classname = m.__class__.__name__
    classname = classname.lower()
    if classname.find('conv') != -1 or classname.find('linear') != -1:
        gain = nn.init.calculate_gain(classname)
        nn.init.xavier_uniform(m.weight, gain=gain)
        if m.bias is not None:
            nn.init.constant(m.bias, 0)
    elif classname.find('batchnorm') != -1:
        nn.init.constant(m.weight, 1)
        if m.bias is not None:
            nn.init.constant(m.bias, 0)
    elif classname.find('embedding') != -1:
        # The default initializer in the TensorFlow embedding layer is a truncated normal with mean 0 and
        # standard deviation 1/sqrt(sparse_id_column.length). Here we use a normal truncated with 3 std dev
        num_columns = m.weight.size(1)
        sigma = 1/(num_columns**0.5)
        m.weight.data.normal_(0, sigma).clamp_(-3*sigma, 3*sigma)

class LRN(nn.Module):

    '''
    Implementing Local Response Normalization layer. Implemention adapted
    from https://github.com/jiecaoyu/pytorch_imagenet/blob/master/networks/model_list/alexnet.py
    '''

    def __init__(self, local_size=1, alpha=1.0, beta=0.75, k=1, ACROSS_CHANNELS=False):
        super(LRN, self).__init__()
        self.ACROSS_CHANNELS = ACROSS_CHANNELS
        if ACROSS_CHANNELS:
            self.average=nn.AvgPool3d(kernel_size=(local_size, 1, 1),
                    stride=1,
                    padding=(int((local_size-1.0)/2), 0, 0))
        else:
            self.average=nn.AvgPool2d(kernel_size=local_size,
                    stride=1,
                    padding=int((local_size-1.0)/2))
        self.alpha = alpha
        self.beta = beta
        self.k = k

    def forward(self, x):
        if self.ACROSS_CHANNELS:
            div = x.pow(2).unsqueeze(1)
            div = self.average(div).squeeze(1)
            div = div.mul(self.alpha).add(self.k).pow(self.beta)
        else:
            div = x.pow(2)
            div = self.average(div)
            div = div.mul(self.alpha).add(self.k).pow(self.beta)
        x = x.div(div)
        return x

class Inception_base(nn.Module):
    def __init__(self, depth_dim, input_size, config):
        super(Inception_base, self).__init__()

        self.depth_dim = depth_dim

        #mixed 'name'_1x1
        self.conv1 = Conv2d_KSE(input_size, config[0][0], kernel_size=1, stride=1, padding=0)

        #mixed 'name'_3x3_bottleneck
        self.conv3_1 = Conv2d_KSE(input_size, config[1][0], kernel_size=1, stride=1, padding=0)
        #mixed 'name'_3x3
        self.conv3_3 = Conv2d_KSE(config[1][0], config[1][1], kernel_size=3, stride=1, padding=1)

        # mixed 'name'_5x5_bottleneck
        self.conv5_1 = Conv2d_KSE(input_size, config[2][0], kernel_size=1, stride=1, padding=0)
        # mixed 'name'_5x5
        self.conv5_5 = Conv2d_KSE(config[2][0], config[2][1], kernel_size=5, stride=1, padding=2)

        self.max_pool_1 = nn.MaxPool2d(kernel_size=config[3][0], stride=1, padding=1)
        #mixed 'name'_pool_reduce
        self.conv_max_1 = Conv2d_KSE(input_size, config[3][1], kernel_size=1, stride=1, padding=0)

        #self.apply(layer_init)

    def forward(self, input):

        output1 = F.relu(self.conv1(input))

        output2 = F.relu(self.conv3_1(input))
        output2 = F.relu(self.conv3_3(output2))

        output3 = F.relu(self.conv5_1(input))
        output3 = F.relu(self.conv5_5(output3))

        output4 = F.relu(self.conv_max_1(self.max_pool_1(input)))

        return torch.cat([output1, output2, output3, output4], dim=self.depth_dim)

# weights available at t https://github.com/antspy/inception_v1.pytorch
class Inception_v1(nn.Module):
    def __init__(self, num_classes=1000):
        super(Inception_v1, self).__init__()

        #conv2d0
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.max_pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.lrn1 = LRN(local_size=11, alpha=0.00109999999404, beta=0.5, k=2)

        #conv2d1
        self.conv2 = Conv2d_KSE(64, 64, kernel_size=1, stride=1, padding=0)

        #conv2d2
        self.conv3  = Conv2d_KSE(64, 192, kernel_size=3, stride=1, padding=1)
        self.lrn3 = LRN(local_size=11, alpha=0.00109999999404, beta=0.5, k=2)
        self.max_pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception_3a = Inception_base(1, 192, [[64], [96,128], [16, 32], [3, 32]]) #3a
        self.inception_3b = Inception_base(1, 256, [[128], [128,192], [32, 96], [3, 64]]) #3b
        self.max_pool_inc3= nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

        self.inception_4a = Inception_base(1, 480, [[192], [ 96,204], [16, 48], [3, 64]]) #4a
        self.inception_4b = Inception_base(1, 508, [[160], [112,224], [24, 64], [3, 64]]) #4b
        self.inception_4c = Inception_base(1, 512, [[128], [128,256], [24, 64], [3, 64]]) #4c
        self.inception_4d = Inception_base(1, 512, [[112], [144,288], [32, 64], [3, 64]]) #4d
        self.inception_4e = Inception_base(1, 528, [[256], [160,320], [32,128], [3,128]]) #4e
        self.max_pool_inc4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception_5a = Inception_base(1, 832, [[256], [160,320], [48,128], [3,128]]) #5a
        self.inception_5b = Inception_base(1, 832, [[384], [192,384], [48,128], [3,128]]) #5b
        self.avg_pool5 = nn.AvgPool2d(kernel_size=7, stride=1, padding=0)

        self.dropout_layer = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, num_classes)

        #self.apply(layer_init)

    def forward(self, input):

        output = self.max_pool1(F.relu(self.conv1(input)))
        output = self.lrn1(output)

        output = F.relu(self.conv2(output))
        output = F.relu(self.conv3(output))
        output = self.max_pool3(self.lrn3(output))

        output = self.inception_3a(output)
        output = self.inception_3b(output)
        output = self.max_pool_inc3(output)

        output = self.inception_4a(output)
        output = self.inception_4b(output)
        output = self.inception_4c(output)
        output = self.inception_4d(output)
        output = self.inception_4e(output)
        output = self.max_pool_inc4(output)

        output = self.inception_5a(output)
        output = self.inception_5b(output)
        output = self.avg_pool5(output)

        output = output.view(-1, 1024)

        if self.fc is not None:
            output = self.dropout_layer(output)
            output = self.fc(output)

        return output


def inception_v1(pretrained=False, checkpoint=None):
    model = Inception_v1(num_classes=1000)
    if pretrained:
        model.load_state_dict(torch.load(checkpoint), strict=False)
    return model




# ==================== Code used to load the weights from torch dump ==========================
ind = {0: 278, 1: 212, 2: 250, 3: 193, 4: 217, 5: 147, 6: 387, 7: 285, 8: 350, 9: 283, 10: 286, 11: 353,
                   12: 334, 13: 150, 14: 249, 15: 362, 16: 246, 17: 166, 18: 218, 19: 172, 20: 177, 21: 148, 22: 357,
                   23: 386, 24: 178, 25: 202, 26: 194, 27: 271, 28: 229, 29: 290, 30: 175, 31: 163, 32: 191, 33: 276,
                   34: 299, 35: 197, 36: 380, 37: 364, 38: 339, 39: 359, 40: 251, 41: 165, 42: 157, 43: 361, 44: 179,
                   45: 268, 46: 233, 47: 356, 48: 266, 49: 264, 50: 225, 51: 349, 52: 335, 53: 375, 54: 282, 55: 204,
                   56: 352, 57: 272, 58: 187, 59: 256, 60: 294, 61: 277, 62: 174, 63: 234, 64: 351, 65: 176, 66: 280,
                   67: 223, 68: 154, 69: 262, 70: 203, 71: 190, 72: 370, 73: 298, 74: 384, 75: 292, 76: 170, 77: 342,
                   78: 241, 79: 340, 80: 348, 81: 245, 82: 365, 83: 253, 84: 288, 85: 239, 86: 153, 87: 185, 88: 158,
                   89: 211, 90: 192, 91: 382, 92: 224, 93: 216, 94: 284, 95: 367, 96: 228, 97: 160, 98: 152, 99: 376,
                   100: 338, 101: 270, 102: 296, 103: 366, 104: 169, 105: 265, 106: 183, 107: 345, 108: 199, 109: 244,
                   110: 381, 111: 236, 112: 195, 113: 238, 114: 240, 115: 155, 116: 221, 117: 259, 118: 181, 119: 343,
                   120: 354, 121: 369, 122: 196, 123: 231, 124: 207, 125: 184, 126: 252, 127: 232, 128: 331, 129: 242,
                   130: 201, 131: 162, 132: 255, 133: 210, 134: 371, 135: 274, 136: 372, 137: 373, 138: 209, 139: 243,
                   140: 222, 141: 378, 142: 254, 143: 206, 144: 186, 145: 205, 146: 341, 147: 261, 148: 248, 149: 215,
                   150: 267, 151: 189, 152: 289, 153: 214, 154: 273, 155: 198, 156: 333, 157: 200, 158: 279, 159: 188,
                   160: 161, 161: 346, 162: 295, 163: 332, 164: 347, 165: 379, 166: 344, 167: 260, 168: 388, 169: 180,
                   170: 230, 171: 257, 172: 151, 173: 281, 174: 377, 175: 208, 176: 247, 177: 363, 178: 258, 179: 164,
                   180: 168, 181: 358, 182: 336, 183: 227, 184: 368, 185: 355, 186: 237, 187: 330, 188: 171, 189: 291,
                   190: 219, 191: 213, 192: 149, 193: 385, 194: 337, 195: 220, 196: 263, 197: 156, 198: 383, 199: 159,
                   200: 287, 201: 275, 202: 374, 203: 173, 204: 269, 205: 293, 206: 167, 207: 226, 208: 297, 209: 182,
                   210: 235, 211: 360, 212: 105, 213: 101, 214: 102, 215: 104, 216: 103, 217: 106, 218: 763, 219: 879,
                   220: 780, 221: 805, 222: 401, 223: 310, 224: 327, 225: 117, 226: 579, 227: 620, 228: 949, 229: 404,
                   230: 895, 231: 405, 232: 417, 233: 812, 234: 554, 235: 576, 236: 814, 237: 625, 238: 472, 239: 914,
                   240: 484, 241: 871, 242: 510, 243: 628, 244: 724, 245: 403, 246: 833, 247: 913, 248: 586, 249: 847,
                   250: 657, 251: 450, 252: 537, 253: 444, 254: 671, 255: 565, 256: 705, 257: 428, 258: 791, 259: 670,
                   260: 561, 261: 547, 262: 820, 263: 408, 264: 407, 265: 436, 266: 468, 267: 511, 268: 609, 269: 627,
                   270: 656, 271: 661, 272: 751, 273: 817, 274: 573, 275: 575, 276: 665, 277: 803, 278: 555, 279: 569,
                   280: 717, 281: 864, 282: 867, 283: 675, 284: 734, 285: 757, 286: 829, 287: 802, 288: 866, 289: 660,
                   290: 870, 291: 880, 292: 603, 293: 612, 294: 690, 295: 431, 296: 516, 297: 520, 298: 564, 299: 453,
                   300: 495, 301: 648, 302: 493, 303: 846, 304: 553, 305: 703, 306: 423, 307: 857, 308: 559, 309: 765,
                   310: 831, 311: 861, 312: 526, 313: 736, 314: 532, 315: 548, 316: 894, 317: 948, 318: 950, 319: 951,
                   320: 952, 321: 953, 322: 954, 323: 955, 324: 956, 325: 957, 326: 988, 327: 989, 328: 998, 329: 984,
                   330: 987, 331: 990, 332: 687, 333: 881, 334: 494, 335: 541, 336: 577, 337: 641, 338: 642, 339: 822,
                   340: 420, 341: 486, 342: 889, 343: 594, 344: 402, 345: 546, 346: 513, 347: 566, 348: 875, 349: 593,
                   350: 684, 351: 699, 352: 432, 353: 683, 354: 776, 355: 558, 356: 985, 357: 986, 358: 972, 359: 979,
                   360: 970, 361: 980, 362: 976, 363: 977, 364: 973, 365: 975, 366: 978, 367: 974, 368: 596, 369: 499,
                   370: 623, 371: 726, 372: 740, 373: 621, 374: 587, 375: 512, 376: 473, 377: 731, 378: 784, 379: 792,
                   380: 730, 381: 491, 382: 7, 383: 8, 384: 9, 385: 10, 386: 11, 387: 12, 388: 13, 389: 14, 390: 15,
                   391: 16, 392: 17, 393: 18, 394: 19, 395: 20, 396: 21, 397: 22, 398: 23, 399: 24, 400: 80, 401: 81,
                   402: 82, 403: 83, 404: 84, 405: 85, 406: 86, 407: 87, 408: 88, 409: 89, 410: 90, 411: 91, 412: 92,
                   413: 93, 414: 94, 415: 95, 416: 96, 417: 97, 418: 98, 419: 99, 420: 100, 421: 127, 422: 128,
                   423: 129, 424: 130, 425: 132, 426: 131, 427: 133, 428: 134, 429: 135, 430: 137, 431: 138, 432: 139,
                   433: 140, 434: 141, 435: 142, 436: 143, 437: 136, 438: 144, 439: 145, 440: 146, 441: 2, 442: 3,
                   443: 4, 444: 5, 445: 6, 446: 389, 447: 391, 448: 0, 449: 1, 450: 390, 451: 392, 452: 393, 453: 396,
                   454: 397, 455: 394, 456: 395, 457: 33, 458: 34, 459: 35, 460: 36, 461: 37, 462: 38, 463: 39, 464: 40,
                   465: 41, 466: 42, 467: 43, 468: 44, 469: 45, 470: 46, 471: 47, 472: 48, 473: 51, 474: 49, 475: 50,
                   476: 52, 477: 53, 478: 54, 479: 55, 480: 56, 481: 57, 482: 58, 483: 59, 484: 60, 485: 61, 486: 62,
                   487: 63, 488: 64, 489: 65, 490: 66, 491: 67, 492: 68, 493: 25, 494: 26, 495: 27, 496: 28, 497: 29,
                   498: 30, 499: 31, 500: 32, 501: 902, 502: 908, 503: 696, 504: 589, 505: 691, 506: 801, 507: 632,
                   508: 650, 509: 782, 510: 673, 511: 545, 512: 686, 513: 828, 514: 811, 515: 827, 516: 583, 517: 426,
                   518: 769, 519: 685, 520: 778, 521: 409, 522: 530, 523: 892, 524: 604, 525: 835, 526: 704, 527: 826,
                   528: 531, 529: 823, 530: 845, 531: 635, 532: 447, 533: 745, 534: 837, 535: 633, 536: 755, 537: 456,
                   538: 471, 539: 413, 540: 764, 541: 744, 542: 508, 543: 878, 544: 517, 545: 626, 546: 398, 547: 480,
                   548: 798, 549: 527, 550: 590, 551: 681, 552: 916, 553: 595, 554: 856, 555: 742, 556: 800, 557: 886,
                   558: 786, 559: 613, 560: 844, 561: 600, 562: 479, 563: 694, 564: 723, 565: 739, 566: 571, 567: 476,
                   568: 843, 569: 758, 570: 753, 571: 746, 572: 592, 573: 836, 574: 714, 575: 475, 576: 807, 577: 761,
                   578: 535, 579: 464, 580: 584, 581: 616, 582: 507, 583: 695, 584: 677, 585: 772, 586: 783, 587: 676,
                   588: 785, 589: 795, 590: 470, 591: 607, 592: 818, 593: 862, 594: 678, 595: 718, 596: 872, 597: 645,
                   598: 674, 599: 815, 600: 69, 601: 70, 602: 71, 603: 72, 604: 73, 605: 74, 606: 75, 607: 76, 608: 77,
                   609: 78, 610: 79, 611: 126, 612: 118, 613: 119, 614: 120, 615: 121, 616: 122, 617: 123, 618: 124,
                   619: 125, 620: 300, 621: 301, 622: 302, 623: 303, 624: 304, 625: 305, 626: 306, 627: 307, 628: 308,
                   629: 309, 630: 311, 631: 312, 632: 313, 633: 314, 634: 315, 635: 316, 636: 317, 637: 318, 638: 319,
                   639: 320, 640: 321, 641: 322, 642: 323, 643: 324, 644: 325, 645: 326, 646: 107, 647: 108, 648: 109,
                   649: 110, 650: 111, 651: 112, 652: 113, 653: 114, 654: 115, 655: 116, 656: 328, 657: 329, 658: 606,
                   659: 550, 660: 651, 661: 544, 662: 766, 663: 859, 664: 891, 665: 882, 666: 534, 667: 760, 668: 897,
                   669: 521, 670: 567, 671: 909, 672: 469, 673: 505, 674: 849, 675: 813, 676: 406, 677: 873, 678: 706,
                   679: 821, 680: 839, 681: 888, 682: 425, 683: 580, 684: 698, 685: 663, 686: 624, 687: 410, 688: 449,
                   689: 497, 690: 668, 691: 832, 692: 727, 693: 762, 694: 498, 695: 598, 696: 634, 697: 506, 698: 682,
                   699: 863, 700: 483, 701: 743, 702: 582, 703: 415, 704: 424, 705: 454, 706: 467, 707: 509, 708: 788,
                   709: 860, 710: 865, 711: 562, 712: 500, 713: 915, 714: 536, 715: 458, 716: 649, 717: 421, 718: 460,
                   719: 525, 720: 489, 721: 716, 722: 912, 723: 825, 724: 581, 725: 799, 726: 877, 727: 672, 728: 781,
                   729: 599, 730: 729, 731: 708, 732: 437, 733: 935, 734: 945, 735: 936, 736: 937, 737: 938, 738: 939,
                   739: 940, 740: 941, 741: 942, 742: 943, 743: 944, 744: 946, 745: 947, 746: 794, 747: 608, 748: 478,
                   749: 591, 750: 774, 751: 412, 752: 771, 753: 923, 754: 679, 755: 522, 756: 568, 757: 855, 758: 697,
                   759: 770, 760: 503, 761: 492, 762: 640, 763: 662, 764: 876, 765: 868, 766: 416, 767: 931, 768: 741,
                   769: 614, 770: 926, 771: 901, 772: 615, 773: 921, 774: 816, 775: 796, 776: 440, 777: 518, 778: 455,
                   779: 858, 780: 643, 781: 638, 782: 712, 783: 560, 784: 433, 785: 850, 786: 597, 787: 737, 788: 713,
                   789: 887, 790: 918, 791: 574, 792: 927, 793: 834, 794: 900, 795: 552, 796: 501, 797: 966, 798: 542,
                   799: 787, 800: 496, 801: 601, 802: 922, 803: 819, 804: 452, 805: 962, 806: 429, 807: 551, 808: 777,
                   809: 838, 810: 441, 811: 996, 812: 924, 813: 619, 814: 911, 815: 958, 816: 457, 817: 636, 818: 899,
                   819: 463, 820: 533, 821: 809, 822: 969, 823: 666, 824: 869, 825: 693, 826: 488, 827: 840, 828: 659,
                   829: 964, 830: 907, 831: 789, 832: 465, 833: 540, 834: 446, 835: 474, 836: 841, 837: 738, 838: 448,
                   839: 588, 840: 722, 841: 709, 842: 707, 843: 925, 844: 411, 845: 747, 846: 414, 847: 982, 848: 439,
                   849: 710, 850: 462, 851: 669, 852: 399, 853: 667, 854: 735, 855: 523, 856: 732, 857: 810, 858: 968,
                   859: 752, 860: 920, 861: 749, 862: 754, 863: 961, 864: 524, 865: 652, 866: 629, 867: 793, 868: 664,
                   869: 688, 870: 658, 871: 459, 872: 930, 873: 883, 874: 653, 875: 768, 876: 700, 877: 995, 878: 549,
                   879: 655, 880: 515, 881: 874, 882: 711, 883: 435, 884: 934, 885: 991, 886: 466, 887: 721, 888: 999,
                   889: 481, 890: 477, 891: 618, 892: 994, 893: 631, 894: 585, 895: 400, 896: 538, 897: 519, 898: 903,
                   899: 965, 900: 720, 901: 490, 902: 854, 903: 905, 904: 427, 905: 896, 906: 418, 907: 430, 908: 434,
                   909: 514, 910: 578, 911: 904, 912: 992, 913: 487, 914: 680, 915: 422, 916: 637, 917: 617, 918: 556,
                   919: 654, 920: 692, 921: 646, 922: 733, 923: 602, 924: 808, 925: 715, 926: 756, 927: 893, 928: 482,
                   929: 917, 930: 719, 931: 919, 932: 442, 933: 563, 934: 906, 935: 890, 936: 689, 937: 775, 938: 748,
                   939: 451, 940: 443, 941: 701, 942: 797, 943: 851, 944: 842, 945: 647, 946: 967, 947: 963, 948: 461,
                   949: 790, 950: 910, 951: 773, 952: 960, 953: 981, 954: 572, 955: 993, 956: 830, 957: 898, 958: 528,
                   959: 804, 960: 610, 961: 779, 962: 611, 963: 728, 964: 759, 965: 529, 966: 419, 967: 929, 968: 885,
                   969: 852, 970: 570, 971: 539, 972: 630, 973: 928, 974: 932, 975: 750, 976: 639, 977: 848, 978: 502,
                   979: 605, 980: 997, 981: 983, 982: 725, 983: 644, 984: 445, 985: 806, 986: 485, 987: 622, 988: 853,
                   989: 884, 990: 438, 991: 971, 992: 933, 993: 702, 994: 557, 995: 504, 996: 767, 997: 824, 998: 959,
                   999: 543}
ind = {val:key for key, val in ind.items()} #actually need the inverse indices

dictionary_default_pytorch_names_to_correct_names_full = {
    'conv1':'conv2d0',
    'conv2':'conv2d1',
    'conv3':'conv2d2',
    'fc':'softmax2'
}

dictionary_default_pytorch_names_to_correct_names_base = {
    'conv1':   'mixed{}_1x1',
    'conv3_1': 'mixed{}_3x3_bottleneck',
    'conv3_3': 'mixed{}_3x3',
    'conv5_1': 'mixed{}_5x5_bottleneck',
    'conv5_5': 'mixed{}_5x5',
    'conv_max_1': 'mixed{}_pool_reduce'
}

def load_weights_from_dump(model, dump_folder):
    # For this to work we need the h5py package
    import h5py
    import numpy as np

    'Loads the weights saved as h5py files in the soumith repo linked above. Just here for completeness'

    dump_folder = os.path.abspath(dump_folder)
    files_list = [os.path.join(dump_folder, x) for x in os.listdir(dump_folder)]

    for name, layer in model.named_parameters():
        # get path from name
        if 'inception' in name:
            first_dot = name.find('.')
            name_inception = name[:first_dot].replace('inception_', '')
            name_layer = name[first_dot + 1:name.find('.', first_dot + 1)]
            name_layer = dictionary_default_pytorch_names_to_correct_names_base[name_layer].format(name_inception)
        else:
            name_layer = name[:name.find('.')]
            name_layer = dictionary_default_pytorch_names_to_correct_names_full[name_layer]
        if 'weight' in name:
            filename = name_layer + '_w.h5'
        else:
            filename = name_layer + '_b.h5'

        filename = os.path.join(dump_folder, filename)
        if filename in files_list:
            files_list.remove(filename)
        else:
            print('file {} not found in files list'.format(filename))

        # print(filename, 'exists', os.path.isfile(filename))

        f = h5py.File(filename, 'r')
        a_group_key = list(f.keys())[0]
        w = np.asarray(list(f[a_group_key]))
        f.close()

        w = torch.from_numpy(w)
        if 'weight' in name:
            w = w.transpose(1, 3).transpose(2, 3).clone()
        w = w.type_as(layer.data)
        if name_layer == 'softmax2':
            #Adjust the size - because google has 1008 classes, class 1 - 1000 are valid
            if 'weight' in name:
                w = w[1:1001, :]
            else:
                w = w[1:1001]

            #and re-arrange the indices - the torch repo had another order for the indices
            ind_list = [ind[idx] for idx in range(1000)]
            idx_t = torch.FloatTensor(ind_list).long()
            w = w.squeeze()
            w = torch.index_select(w, dim=0, index=idx_t)

        if layer.data.size() != w.size():
            raise ValueError('Incompatible sizes')

        layer.data = w

    print('Number of unused files: {}'.format(len(files_list)))
    return model
