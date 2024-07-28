import { Card, CardContent, CardHeader, CardTitle } from "../ui/card";
import { useEffect, useState } from "react";

export const SideNav = ({ logitsExist }: { logitsExist: boolean }) => {
  const [activeId, setActiveId] = useState("");
  let idList;
  if (logitsExist) {
    idList = ["Top", "Hist.", "Logits", "Act."];
  } else {
    idList = ["Top", "Hist.", "Act."];
  }

  const handleScroll = () => {
    const sections = document.querySelectorAll("div[id]");
    let currentSectionId = "";

    sections.forEach((section) => {
      if (idList.indexOf(section.id) != -1) {
        const rect = section.getBoundingClientRect();
        if (rect.top <= window.innerHeight / 2) {
          currentSectionId = section.id;
        }
      }
    });

    setActiveId(currentSectionId);
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
    <Card className="side-nav py-4">
      <CardHeader className="p-0">
        <CardTitle className="flex justify-between items-center text-xs p-2">
          <span className="font-bold">CONTENTS</span>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="flex flex-col">
          <ul>
            {idList.map((item) => (
              <li key={item} style={{ position: "relative", marginBottom: "10px" }}>
                <a
                  href={"#" + item}
                  style={{
                    textDecoration: "none",
                    display: "block",
                    padding: "10px 0",
                    color: activeId === item ? "blue" : "black",
                  }}
                >
                  {item}
                </a>
                {activeId === item && (
                  <div
                    style={{
                      position: "absolute",
                      left: "-10px",
                      top: "0",
                      bottom: "0",
                      width: "2px",
                      backgroundColor: "blue",
                    }}
                  ></div>
                )}
              </li>
            ))}
          </ul>
        </div>
      </CardContent>
    </Card>
  );
};
