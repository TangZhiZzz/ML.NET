using Microsoft.ML;
using Microsoft.ML.Data;
using System;

class QnAData
{
    [LoadColumn(0)]
    public string Question { get; set; }

    [LoadColumn(1)]
    public string Answer { get; set; }
}

class Program
{
    static void Main(string[] args)
    {
        // 创建MLContext对象
        MLContext mlContext = new MLContext();

        // 加载数据集
        string dataPath = @"daily_conversation_dataset.csv";
        IDataView qnaDataView = mlContext.Data.LoadFromTextFile<QnAData>(dataPath, separatorChar: ',');

        // 数据预处理
        var dataProcessingPipeline = mlContext.Transforms.Conversion.MapValueToKey("Label", "Answer")
            .Append(mlContext.Transforms.Text.FeaturizeText("Features", "Question"))
            .Append(mlContext.Transforms.Concatenate("Input", "Features"))
            .Append(mlContext.Transforms.NormalizeMinMax("Input"));

        IDataView data = dataProcessingPipeline.Fit(qnaDataView).Transform(qnaDataView);

        // 选择算法模型
        var trainingPipeline = mlContext.Transforms.Conversion.MapValueToKey("Label", "Answer")
            .Append(mlContext.Transforms.Text.FeaturizeText("Features", "Question"))
            .Append(mlContext.Transforms.Concatenate("Input", "Features"))
            .Append(mlContext.Transforms.NormalizeMinMax("Input"))
            .Append(mlContext.MulticlassClassification.Trainers.SdcaNonCalibrated(labelColumnName: "Label", featureColumnName: "Input"))
            .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

        // 训练模型
        ITransformer trainedModel = trainingPipeline.Fit(data);


        // 保存模型
        mlContext.Model.Save(trainedModel, data.Schema, "QnAModel.zip");

        // 加载模型
        ITransformer loadedModel;

        try
        {
            loadedModel = mlContext.Model.Load("QnAModel.zip", out var schema);
        }
        catch (System.IO.InvalidDataException e)
        {
            Console.WriteLine("模型的架构与加载的数据不兼容");
            return;
        }


        // 预测问题的答案
        var predictor = mlContext.Model.CreatePredictionEngine<QnAData, QnAPrediction>(loadedModel);
        while (true)
        {
            Console.Write("请输入问题：");
            var question = Console.ReadLine();
            var prediction = predictor.Predict(new QnAData { Question = question });
            Console.WriteLine("答案是：" + prediction.Answer);
        }


    }
    class QnAPrediction
    {
        [ColumnName("PredictedLabel")]
        public string Answer { get; set; }
    }
}