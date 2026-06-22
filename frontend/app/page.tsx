import Link from "next/link";
import { InteractiveRobotSpline } from "@/components/ui/interactive-3d-robot";
import { StackedCircularFooter } from "@/components/ui/stacked-circular-footer";
import { Button } from "@/components/ui/button";

export default function Home() {
  return (
    <main>
      <div className="relative w-screen h-screen overflow-hidden">
        <InteractiveRobotSpline
          scene="https://prod.spline.design/PyzDhpQ9E5f1E3MT/scene.splinecode"
          className="absolute inset-0 z-0"
        />
        <div className="absolute inset-0 z-10 pt-20 md:pt-32 px-4 pointer-events-none">
          <div className="text-center text-white drop-shadow-lg max-w-2xl mx-auto">
            <h1 className="text-3xl md:text-5xl font-bold">
              Doubts resolved instantly, your way.
            </h1>
            <p className="mt-4 text-lg text-zinc-200">
              Curriculum-aligned answers from your own institute&apos;s material.
            </p>
            <div className="mt-8 pointer-events-auto flex gap-3 justify-center">
              <Link href="/chat">
                <Button size="lg">Ask a doubt</Button>
              </Link>
              <Link href="/admin">
                <Button size="lg" variant="outline" className="bg-white/10 text-white border-white/30">
                  Teacher dashboard
                </Button>
              </Link>
            </div>
          </div>
        </div>
      </div>
      <StackedCircularFooter />
    </main>
  );
}
