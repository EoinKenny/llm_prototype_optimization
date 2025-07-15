prototypes_dict = {
    'dbpedia': [
        # Athlete
        ("The marathon runner crossed the finish line with a record time.", 0, "Athlete"),
        ("The sprinter trained for years to perfect their start technique.", 0, "Athlete"),
        ("An inspiring performance by the gymnast left the crowd cheering.", 0, "Athlete"),
        # Person
        ("The author published their memoir to critical acclaim.", 1, "Person"),
        ("She is renowned for her groundbreaking research in physics.", 1, "Person"),
        ("He dedicated his life to humanitarian work in underserved communities.", 1, "Person"),
        # Animal
        ("The rare bird species builds its nest high in the canopy.", 2, "Animal"),
        ("A herd of elephants migrates across the savannah each year.", 2, "Animal"),
        ("The nocturnal creature hunts quietly under the cover of darkness.", 2, "Animal"),
        # Building
        ("The historic cathedral features Gothic spires and stained glass windows.", 3, "Building"),
        ("A modern skyscraper dominates the city skyline at dusk.", 3, "Building"),
        ("The renovated library now includes eco-friendly solar panels.", 3, "Building"),
        # Politician
        ("The senator introduced a bill aiming to reform the healthcare system.", 4, "Politician"),
        ("A seasoned politician addressed the crowd with promises of change.", 4, "Politician"),
        ("The mayor launched a new initiative to revitalize downtown business.", 4, "Politician"),
        # Company
        ("The startup unveiled an AI-driven platform for better customer insights.", 5, "Company"),
        ("The multinational corporation announced layoffs amid restructuring.", 5, "Company"),
        ("A small tech firm secured venture capital funding for expansion.", 5, "Company"),
        # Organisation
        ("The charity organized a fundraising gala to support local shelters.", 6, "Organisation"),
        ("An international NGO coordinated relief efforts after the earthquake.", 6, "Organisation"),
        ("The committee set new guidelines for ethical research practices.", 6, "Organisation"),
        # MusicalWork
        ("The symphony premiered to a sold-out audience at the concert hall.", 7, "MusicalWork"),
        ("The jazz quartet released a new album blending classic and modern styles.", 7, "MusicalWork"),
        ("A haunting melody from the opera moved the audience to tears.", 7, "MusicalWork"),
        # WinterSportPlayer
        ("The ice skater executed a flawless triple axel.", 8, "WinterSportPlayer"),
        ("A top-ranked snowboarder landed a new trick in competition.", 8, "WinterSportPlayer"),
        ("The bobsled team set a track record during the Olympic trials.", 8, "WinterSportPlayer"),
        # SocietalEvent
        ("A peaceful march was held to raise awareness for climate action.", 9, "SocietalEvent"),
        ("The annual cultural festival drew thousands of attendees downtown.", 9, "SocietalEvent"),
        ("A commemorative ceremony honored veterans of past conflicts.", 9, "SocietalEvent"),
    ],
    '20newsgroups': [
        # alt.atheism
        ("Many debate the philosophical arguments against the existence of deities.", 0, "alt.atheism"),
        ("A spirited discussion about secularism took place online.", 0, "alt.atheism"),
        ("She argued that morality can exist without religious belief.", 0, "alt.atheism"),
        # comp.graphics
        ("The latest graphics card supports real-time ray tracing.", 1, "comp.graphics"),
        ("He shared a tutorial on creating textures in the 3D engine.", 1, "comp.graphics"),
        ("The rendering algorithm improved frame rates significantly.", 1, "comp.graphics"),
        # comp.os.ms-windows.misc
        ("A known issue causes the system to freeze after a Windows update.", 2, "comp.os.ms-windows.misc"),
        ("Tips for optimizing memory usage on older Windows machines.", 2, "comp.os.ms-windows.misc"),
        ("The registry tweak resolved the driver conflict successfully.", 2, "comp.os.ms-windows.misc"),
        # comp.sys.ibm.pc.hardware
        ("Upgrading the CPU cache boosted performance noticeably.", 3, "comp.sys.ibm.pc.hardware"),
        ("He replaced the faulty power supply in his vintage IBM PC.", 3, "comp.sys.ibm.pc.hardware"),
        ("Troubleshooting RAM errors with memory diagnostics tools.", 3, "comp.sys.ibm.pc.hardware"),
        # comp.sys.mac.hardware
        ("A new firmware update enhanced battery life on the laptop.", 4, "comp.sys.mac.hardware"),
        ("She installed additional SSD storage in her Mac Mini.", 4, "comp.sys.mac.hardware"),
        ("Resetting the SMC fixed the keyboard backlight issue.", 4, "comp.sys.mac.hardware"),
        # comp.windows.x
        ("Configuring the X server allowed for multiple displays.", 5, "comp.windows.x"),
        ("Using xrandr, he adjusted the screen resolution on the fly.", 5, "comp.windows.x"),
        ("A custom window manager theme improved the desktop aesthetics.", 5, "comp.windows.x"),
        # misc.forsale
        ("Selling a barely used DSLR camera with extra lenses.", 6, "misc.forsale"),
        ("Looking to buy a second-hand guitar in good condition.", 6, "misc.forsale"),
        ("Offering vintage vinyl records at reasonable prices.", 6, "misc.forsale"),
        # rec.autos
        ("The new electric vehicle boasts over 300 miles of range.", 7, "rec.autos"),
        ("He detailed the maintenance schedule for his classic car.", 7, "rec.autos"),
        ("An in-depth review of the latest sports coupe.", 7, "rec.autos"),
        # rec.motorcycles
        ("A touring bike is ideal for long-distance rides.", 8, "rec.motorcycles"),
        ("He installed heated grips for winter motorcycle commutes.", 8, "rec.motorcycles"),
        ("Comparing the engine specs of two sportbike models.", 8, "rec.motorcycles"),
        # rec.sport.baseball
        ("The pitcher threw a no-hitter in last night's game.", 9, "rec.sport.baseball"),
        ("A late-game home run secured the championship title.", 9, "rec.sport.baseball"),
        ("Analyzing batting averages of top hitters this season.", 9, "rec.sport.baseball"),
    ],
    'trec': [
        # animal (index 2)
        ("Which animal can run the fastest over long distances?", 2, "animal"),
        ("What mammal is known for its exceptional memory?", 2, "animal"),
        ("How do migratory birds navigate during their long journeys?", 2, "animal"),
        # color (index 19)
        ("What color do you get by mixing red and blue paint?", 19, "color"),
        ("Which color is most commonly associated with royalty?", 19, "color"),
        ("How does the sky appear at sunset in terms of color?", 19, "color"),
        # country (index 18)
        ("Which country has the highest number of UNESCO World Heritage sites?", 18, "country"),
        ("What country is both a city and a sovereign state?", 18, "country"),
        ("Which country is the largest by land area?", 18, "country"),
        # city (index 21)
        ("Which city is known as the Eternal City?", 21, "city"),
        ("What city is the capital of Japan?", 21, "city"),
        ("Which city hosts the annual Carnival festival in Brazil?", 21, "city"),
        # date (index 8)
        ("When did the Berlin Wall fall?", 8, "date"),
        ("On what date did the first man land on the moon?", 8, "date"),
        ("When is Thanksgiving celebrated in the United States?", 8, "date"),
        # event (index 10)
        ("What event triggered the start of World War I?", 10, "event"),
        ("Which event marks the beginning of the Renaissance period?", 10, "event"),
        ("What event is commemorated on July 14th in France?", 10, "event"),
        # food (index 17)
        ("What food is traditionally used to make sushi?", 17, "food"),
        ("Which food item is the main ingredient in guacamole?", 17, "food"),
        ("What food group does lentils belong to?", 17, "food"),
        # money (index 25)
        ("How much does a gallon of gasoline cost on average?", 25, "money"),
        ("What is the currency used in Switzerland?", 25, "money"),
        ("How much did the new smartphone retail for at launch?", 25, "money"),
        # sport (index 29)
        ("Which sport uses a shuttlecock instead of a ball?", 29, "sport"),
        ("What sport features events like slalom and giant slalom?", 29, "sport"),
        ("Which sport is known as the 'king of sports'?", 29, "sport"),
        # temp (index 41)
        ("What is the average temperature of the Earth's core?", 41, "temp"),
        ("At what temperature does water boil at sea level?", 41, "temp"),
        ("What is the normal human body temperature in Celsius?", 41, "temp"),
    ]
}
