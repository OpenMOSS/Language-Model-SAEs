import { cn } from "@/lib/utils";
import { Card, CardContent, CardHeader, CardTitle } from "../ui/card";
import { useEffect, useState } from "react";

export const SectionNavigator = ({ sections }: { sections: { title: string; id: string }[] }) => {
  const [activeSection, setActiveSection] = useState<{ title: string; id: string } | null>(null);

  const handleScroll = () => {
    // Use reduce instead of find for obtaining the last section that is in view
    const currentSection = sections.reduce((result: { title: string; id: string } | null, section) => {
      const secElement = document.getElementById(section.id);
      if (!secElement) return result;
      const rect = secElement.getBoundingClientRect();
      if (rect.top <= window.innerHeight / 2) {
        return section;
      }
      return result;
    }, null);

    setActiveSection(currentSection);
  };

  useEffect(() => {
    window.addEventListener("scroll", handleScroll);

    // Run the handler to set the initial active section
    handleScroll();

    return () => {
      window.removeEventListener("scroll", handleScroll);
    };
  });

  return (
    <Card className="py-4 sticky top-0 w-60 h-full bg-transparent">
      <CardHeader className="py-0">
        <CardTitle className="flex justify-between items-center text-xs p-2">
          <span className="font-bold">CONTENTS</span>
        </CardTitle>
      </CardHeader>
      <CardContent className="py-0">
        <div className="flex flex-col">
          <ul>
            {sections.map((section) => (
              <li key={section.id} className="relative">
                <a
                  href={"#" + section.id}
                  className={cn("p-2 block text-neutral-700", activeSection === section && "text-[blue]")}
                >
                  {section.title}
                </a>
                {activeSection === section && <div className="absolute -left-1.5 top-0 bottom-0 w-0.5 bg-[blue]"></div>}
              </li>
            ))}
          </ul>
        </div>
      </CardContent>
    </Card>
  );
};
