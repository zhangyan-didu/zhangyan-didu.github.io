import type {ReactNode} from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import HomepageFeatures from '@site/src/components/HomepageFeatures';
import Heading from '@theme/Heading';
import { useState } from 'react';

import styles from './index.module.css';

const poetryList = [
  "一枪茶。二旗茶。休献机心名利家。无眠为作差。",
  "三杯通大道，一斗合自然。",
  "一樽齐死生，万事固难审。",
  "两人对酌山花开，一杯一杯复一杯。",
  "万籁此都寂，但余钟磬音。",
  "坐酌泠泠水，看煎瑟瑟尘。",
  "我来问道无馀说，云在青霄水在瓶。",
  "溪花与禅意，相对亦忘言。"
];

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  const [currentPoetry, setCurrentPoetry] = useState<string>('');

  const handleTeaClick = () => {
    const randomIndex = Math.floor(Math.random() * poetryList.length);
    setCurrentPoetry(poetryList[randomIndex]);
  };

  return (
    <header className={clsx('hero hero--primary', styles.heroBanner)}>
      <div className="container">
        <Heading as="h1" className="hero__title">
          {siteConfig.title}
        </Heading>
        <p className="hero__subtitle">{siteConfig.tagline}</p>
        <div className={styles.buttons}>
          <button
            className="button button--secondary button--lg"
            onClick={handleTeaClick}
          >
            Welcome
          </button>
        </div>
        {currentPoetry && (
          <div className={styles.poetryDisplay}>
            <p className={styles.poetryText}>{currentPoetry}</p>
          </div>
        )}
      </div>
    </header>
  );
}

export default function Home(): ReactNode {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title={`Hello from ${siteConfig.title}`}
      description="Description will go into a meta tag in <head />">
      <HomepageHeader />
      <main>
        <HomepageFeatures />
      </main>
    </Layout>
  );
}
