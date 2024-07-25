if __name__ == "__main__":

    from minigpt4.common.config import Config
    from train import parse_args
    import minigpt4.tasks as tasks
    
    args = parse_args()
    cfg = Config(args)
    
    task = tasks.setup_task(cfg)
    
    datasets = task.build_datasets(cfg)
    
    dataset = datasets['mimic_cxr']['train']

    visual_processor = dataset.vis_processor
    text_processor = dataset.text_processor
    
    img_id = '5da6598b-f5a17c40-e9bf1a88-b9747692-9f721582'
    
    image = dataset.load_image(img_id)
    import random
    instruction = random.choice(dataset.instruction_pool)
    instruction = f'<Img><ImageHere></Img> {text_processor(instruction)}'
    image = image.unsqueeze(0)
    
    demo_sample = {
        "image": image,
        "instruction_input": instruction,
        "answer": "Bilateral nodular opacities, which most likely represent nipple shadows, are observed. There is no focal consolidation, pleural effusion, or pneumothorax. Cardiomediastinal silhouette is normal, and there is no acute cardiopulmonary process. Clips project over the left lung, potentially within the breast, and the imaged upper abdomen is unremarkable. Chronic deformity of the posterior left sixth and seventh ribs is noted.",
        "image_id": img_id,
    }
    
    print(demo_sample)
    
    model = task.build_model(cfg)
    print(model)
    output = model(demo_sample)