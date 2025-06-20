from excel_codes.my_excel import myExcel as EX
import numpy as np
import matplotlib.pyplot as plt

def create_bar_gaph(excelFileName, sheetNames, x):

    excel = EX(excelFileName, readWrite = 'read')
    excel.read_excel(sheetNames)

    def autolabel(rects, ss):
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2., 1.01*height,
                    ss % height,
                    ha='center', va='bottom')

    rows = excel.get_rows()
    cols = excel.get_cols()
    models = [x for x in sorted(rows['PSNR'])]
    cases = [x.split('_')[0] for x in sorted(cols['PSNR'])]
    numCasesPerFold = 4
    folds = np.arange(0,int(len(cases)/numCasesPerFold)) + 1
    foldsLabels = ['Fold_' + str(x) for x in folds]

    real = {}
    syn = {}
    real['psnr'] = {}
    syn['psnr'] = {}
    real['ssim'] = {}
    syn['ssim'] = {}

    for i,row in enumerate(models):
        psnrRow = np.array(rows['PSNR'][row])
        ssimRow = np.array(rows['SSIM'][row])
        modelName = row.split('_')[0]

        if(not modelName in syn['psnr']):
            syn['psnr'][modelName] = []
            syn['ssim'][modelName] = []
        if (not modelName in real['psnr']):
            real['psnr'][modelName] = []
            real['ssim'][modelName] = []

        foldIdx = np.mod(np.int(i / 2), len(folds))
        inds = foldInds[foldIdx]
        psnrFold = psnrRow[inds]
        psnrFold = np.mean(psnrFold)
        ssimFold = ssimRow[inds]
        ssimFold = np.mean(ssimFold)
        if(np.mod(i,2) == 0):
            real['psnr'][modelName].append(psnrFold)
            real['ssim'][modelName].append(ssimFold)
        else:
            syn['psnr'][modelName].append(psnrFold)
            syn['ssim'][modelName].append(ssimFold)

    modelsToShow = ['CHEN', 'DnCnn', 'REDNET', 'perceptualNet']
    qualityIndices = ['psnr', 'ssim']
    gapToShow = {'ssim' : 0.1, 'psnr' : 5}
    ss = {'ssim' : '%0.2f','psnr' : '%0.1f'}
    modelNamesForArticle = ['LD-CNN', 'DnCNN', 'RED-CNN', 'CNN-VGG']

    for kind in qualityIndices:
        for i,modelToShow in enumerate(modelsToShow):
            fig, ax = plt.subplots()
            width = 0.35
            p1 = ax.bar(folds, real[kind][modelToShow], width = width, color='b', bottom=0, align = 'edge')
            p2 = ax.bar(folds + width, syn[kind][modelToShow], width = width, color='r', bottom=0, align = 'edge')
            ax.set_title(modelNamesForArticle[i])
            ax.set_ylabel(kind.upper())
            minToShow = np.min((np.min(real[kind][modelToShow]), np.min(syn[kind][modelToShow]))) - gapToShow[kind]
            maxToShow = np.max((np.max(real[kind][modelToShow]), np.max(syn[kind][modelToShow]))) + gapToShow[kind]
            ax.set_ylim([minToShow, maxToShow])
            ax.set_xticks(folds + width / 2)
            ax.set_xticklabels(foldsLabels, rotation = 17)
            ax.legend((p1[0], p2[0]), ('Real', 'Synthetic'))
            autolabel(p1, ss[kind])
            autolabel(p2, ss[kind])

    plt.show()